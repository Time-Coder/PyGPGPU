"""Random draws in a kernel body: the spellings, and the state one work item draws from.

A body writes the call it already knows -- ``random.random()``, ``np.random.uniform(0,
1)`` -- and this turns it into one of the functions ``include/random.h`` declares. The
work item and the seed are the launch's business (see ``kernel/rng.py``), so a body never
passes either, and a device helper that draws receives its caller's state by pointer the
way the exception slot already travels (see ``_emit_device_call``).

Two rules keep the emitted code honest. The *object* the callee resolves to decides
whether a call is a draw, so a user's own function called ``random`` is theirs; and the
state is declared once at the top of the kernel body, so every draw in the body -- and in
every helper it reaches -- advances one stream in source order. That is what makes the
sequence reproducible: the k-th draw of a work item is the same number whatever order the
work items actually run in.
"""
from __future__ import annotations

import ast
import types
from typing import List, Optional, Tuple

from .. import rng as random_support
from ..device_builtins import emit_index
from .notation import Translated
from .predicates import is_complex, is_image, is_ptr
from .vectors import is_vec


class _RandomEmitMixin:
    """The random-draw vocabulary, its per-work-item state, and their declarations."""

    # The per-work-item stream a kernel declares once and every draw advances.
    RANDOM_STATE: str = "_pyg_rng"

    # The launch's seed: a scalar the kernel signature declares and the host binds.
    RANDOM_SEED: str = "_pyg_random_seed"

    # Class-level defaults, so a translator that never draws answers without init work --
    # the shape `_RenderMixin.has_barrier` uses. Both are set on the instance the first
    # time a draw is emitted.
    uses_random: bool = False
    random_needs_double: bool = False


    def _random_callee(self, node: ast.expr) -> Optional[object]:
        """The object a call names, when it is reached through a module or a class.

        Only a module (or a class) is walked, and only one attribute deep: `random.random`
        and `np.random.random` are module attributes, while `self.x.y()` is a value's member
        and belongs to the method-on-value path. Restricting the walk also keeps a property
        from being read for an attribute that could never be a draw.

        A name the *body* binds is not a module: a local `random = 3.0` shadows the import,
        and `random.random()` there is an attribute of a float, which python would refuse.
        The translator's own `_local_names` answers that, so the module is the import and
        never the local -- and a name the body binds is not walked at all.
        """
        if isinstance(node, ast.Name):
            if node.id in self._local_names():
                return None
            return self.resolve_global(node.id)
        if isinstance(node, ast.Attribute):
            owner: Optional[object] = self._random_callee(node.value)
            if not isinstance(owner, (types.ModuleType, type)):
                return None
            return getattr(owner, node.attr, None)
        return None


    def _emit_random_call(self, node: ast.Call) -> Optional[Translated]:
        """Translate a random draw, or None when the call is not one.

        Recognition is by the object the callee resolves to, the rule every builtin
        follows: the table holds python's and numpy's own callables, because those are the
        objects a body's ``random.random()`` resolves to.
        """
        callee: Optional[object] = self._random_callee(node.func)
        if callee is None:
            return None
        spelling: Optional[random_support.RandomSpelling] = (
            random_support.random_spelling(callee))
        if spelling is None:
            self._refuse_unsupported_random(node)
            return None
        return self._emit_random_draw(spelling[0], spelling[1], node)


    def _refuse_unsupported_random(self, node: ast.Call) -> None:
        """Name the reason when the callee is one of the random modules' other functions.

        Without this a body calling `np.random.seed(0)` would be reported as "not a python
        function this translator can compile", which says nothing about why a kernel cannot
        seed, or what to write instead.
        """
        callee: ast.expr = node.func
        if not isinstance(callee, ast.Attribute):
            return
        owner: Optional[object] = self._random_callee(callee.value)
        if not random_support.is_random_module(owner):
            return
        spelling: str = (f"{random_support.module_spelling(owner)}.{callee.attr}")
        self._error(
            f"{spelling}() has no device meaning inside a kernel: a work item draws "
            f"single values from the launch's own stream, and seeding is the launch's. "
            f"The draws a kernel body may write are "
            f"{', '.join(random_support.supported_spellings())}", node)


    def _emit_random_draw(self, operation: str, counts: Tuple[int, ...],
                          node: ast.Call) -> Translated:
        """One draw: the header function for the operation, at the draw's width."""
        self._check_random_arguments(counts, node)
        args: List[Translated] = [self._visit_expression(argument) for argument in node.args]
        for _code, t in args:
            if is_ptr(t) or is_image(t) or is_vec(t) or is_complex(t):
                self._error(
                    f"a random draw is one scalar number per call, and this argument is "
                    f"{t}: draw a scalar and use it per component", node)
        state: str = self._random_state()
        if operation == random_support.UNIFORM:
            width: str = self._random_width([t for _code, t in args])
            self._require_random_width(width)
            function: str = ("pyg_rng_uniform_double" if width == "double"
                             else "pyg_rng_uniform")
            return f"{function}({state})", width
        if operation == random_support.RANGE:
            width = self._random_width([t for _code, t in args])
            self._require_random_width(width)
            low: str = args[0][0] if len(args) == 2 else "0"
            high: str = args[1][0] if len(args) == 2 else "1"
            function = ("pyg_rng_range_double" if width == "double"
                        else "pyg_rng_range")
            return (f"{function}({state}, ({width})({low}), ({width})({high}))", width)
        if operation == random_support.NORMAL:
            width = self._random_width([t for _code, t in args])
            self._require_random_width(width)
            mean: str = args[0][0] if len(args) == 2 else "0"
            deviation: str = args[1][0] if len(args) == 2 else "1"
            function = ("pyg_rng_normal_double" if width == "double"
                        else "pyg_rng_normal")
            return (f"{function}({state}, ({width})({mean}), ({width})({deviation}))",
                    width)
        # The two integer draws: `pyg_rng_int` is [low, high) -- numpy's `randint` and
        # python's `randrange` -- and `pyg_rng_int_inclusive` is python's `randint`, which
        # includes its high end. One argument means "from zero up to it".
        low_code: str = args[0][0] if len(args) == 2 else "0"
        high_index: int = 1 if len(args) == 2 else 0
        high_code: str = args[high_index][0]
        function = ("pyg_rng_int_inclusive" if operation == random_support.INT_INCLUSIVE
                    else "pyg_rng_int")
        return f"{function}({state}, (int)({low_code}), (int)({high_code}))", "int"


    def _check_random_arguments(self, counts: Tuple[int, ...], node: ast.Call) -> None:
        """Validate the call's shape: positional only, and one of the accepted counts."""
        for keyword in node.keywords:
            name: str = keyword.arg or "**"
            if name == "size":
                self._error(
                    "a random draw answers one number per call, so a `size=` (or a shape "
                    "argument) cannot be represented in a kernel body: it would build an "
                    "array, which is host work. Draw one value and store it in the element "
                    "that needs it", node)
            self._error(
                f"a random draw takes positional arguments only, not '{name}': write the "
                f"call as random.uniform(0.0, 1.0)", node)
        provided: int = len(node.args)
        if provided in counts:
            return
        if provided > max(counts):
            self._error(
                f"a random draw answers one number per call, and this call names "
                f"{provided} argument{'' if provided == 1 else 's'}: a size or a shape "
                f"would make it build an array, which a kernel body cannot represent. "
                f"Draw one value and store it in the element that needs it", node)
        self._error(
            f"this random draw takes {_counts_text(counts)} argument(s), got {provided}",
            node)


    def _random_width(self, types: List[object]) -> str:
        """The width one draw answers: the kernel's float width, widened by an argument.

        A draw answers a python float, so it follows the same mapping a float literal does
        (`config.type_map`); a double argument widens it the way it widens any other call.
        MSL has no `double` at all, so Metal answers the float32 form -- the same narrowing
        `_emit_math_call` applies to a numpy call with integer arguments.
        """
        if self._metal_syntax:
            return "float"
        width: str = self.type_map.get("float", "float")
        if width == "double" or "double" in types:
            return "double"
        return "float"


    def _require_random_width(self, width: str) -> None:
        """Record that this unit draws, and at which width the header must compile."""
        self.uses_random = True
        if width == "double":
            self.random_needs_double = True


    def _random_state(self) -> str:
        """The state expression one draw passes: a kernel's own state, a helper's pointer."""
        self.uses_random = True
        return f"&{self.RANDOM_STATE}" if self.is_kernel else self.RANDOM_STATE


    def random_seed_parameter(self) -> List[str]:
        """The launch-seed parameter a kernel that draws declares."""
        if not self.uses_random or not self.is_kernel:
            return []
        return [self._scalar_param("uint", self.RANDOM_SEED)]


    def random_state_parameter(self) -> List[str]:
        """The state pointer a device helper that draws receives from its caller."""
        if not self.uses_random or self.is_kernel:
            return []
        # MSL wants an address space on a pointer parameter, and the state a caller hands
        # over lives in its thread-local storage; OpenCL's unqualified pointer parameter is
        # already `__private`, and CUDA's is a plain pointer into local memory.
        space: str = "thread " if self._metal_syntax else ""
        return [f"{space}pyg_rng* {self.RANDOM_STATE}"]


    def random_entry_lines(self) -> List[str]:
        """The state a kernel declares once per invocation, and its seeding.

        The lines go at the top of the body, before the magic loop's head: a helper later
        in the body takes `&<state>`, so the state has to exist by then. Each work item's
        stream is seeded from the launch seed and that work item's own position along the
        three launch axes, which is what makes two work items two streams.
        """
        if not self.uses_random or not self.is_kernel:
            return []
        # The three launch axes, folded into the stream id. `emit_index` already spells the
        # backend's own form (an intrinsic on OpenCL, `blockIdx`/`threadIdx` arithmetic on
        # CUDA, the prelude's macro on Metal), and the header's parameter is `unsigned int`,
        # so the implicit conversion is the whole of it.
        coordinates: str = ", ".join(
            emit_index("get_global_id", axis, self.backend) for axis in range(3))
        return [f"    pyg_rng {self.RANDOM_STATE};",
                f"    pyg_rng_init(&{self.RANDOM_STATE}, {self.RANDOM_SEED}, {coordinates});"]


def _counts_text(counts: Tuple[int, ...]) -> str:
    """The accepted argument counts, spelled for an error message."""
    return " or ".join(str(count) for count in counts)