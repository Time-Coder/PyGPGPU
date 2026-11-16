from __future__ import annotations
import ast
import enum
import math
from typing import (Any, Dict, List, Optional, Tuple)
import numpy as np
from ..device_builtins import (
    INDEX_VECTOR_BUILTINS,
    INDEX_VECTOR_OPENCL,
)
from .notation import (
    BLOCK_ARRAY,
    DeviceType,
    HostValueEntry,
    PTR,
    Translated,
    device_name,
)
from .receivers import (
    _enum_member_value,
)
from .predicates import (
    promote,
)
from .type_map import (
    _check_integer_fits,
)


class _ExpressionsMixin:

    def _visit_expression(self, node:ast.expr)->Translated:
        if isinstance(node, ast.Constant):
            return self._visit_constant(node)
        if isinstance(node, ast.Name):
            return self._visit_name(node)
        if isinstance(node, ast.BinOp):
            return self._visit_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._visit_unaryop(node)
        if isinstance(node, ast.Compare):
            return self._visit_compare(node)
        if isinstance(node, ast.BoolOp):
            # Python's `and`/`or` answer an **operand**, not a bool: `2.0 and 3.0` is 3.0 and
            # `-5.0 or 3.0` is -5.0 (it is the truthiness that is tested, and the value that
            # is returned). Emitting `&&`/`||` answered `1.0` for both, so a body using the
            # result as a number (`(a and b) * 10.0`) computed with 1.0 where python used
            # the operand -- a silent wrong answer, and one the condition position never
            # showed because there only the truth value is read.
            #
            # The value form is the conditional operator python's own semantics reduce to:
            # `a and b` is `a ? b : a`, `a or b` is `a ? a : b`. The test is spelled as an
            # explicit comparison rather than the bare code: OpenCL C refuses a float as a
            # conditional's condition ("used type 'float' where floating point type is not
            # allowed"), while CUDA accepts it -- so a bare float test compiled on CUDA and
            # failed on OpenCL for the same source. A comparison is a bool on both.
            codes:List[Translated] = [self._visit_expression(v) for v in node.values]
            parts:List[str] = [self._truth_of(code, t) for code, t in codes]
            is_and:bool = isinstance(node.op, ast.And)
            # Fold from the right: `a and b and c` is `a ? (b ? c : b) : a`.
            folded_code, folded_t = codes[-1]
            for index in range(len(codes) - 2, -1, -1):
                _test_code, left_t = codes[index]
                test:str = parts[index]
                left_code = _test_code
                if is_and:
                    then_code, else_code = folded_code, left_code
                else:
                    then_code, else_code = left_code, folded_code
                if left_t != folded_t:
                    unified:object = promote(left_t, folded_t)
                    folded_code = f"({test} ? ({unified})({then_code}) : ({unified})({else_code}))"
                    folded_t = unified
                else:
                    folded_code = f"({test} ? {then_code} : {else_code})"
            return folded_code, folded_t
        if isinstance(node, ast.IfExp):
            cond:str = self._condition_code(node.test)
            then_code, tt = self._visit_expression(node.body)
            else_code, et = self._visit_expression(node.orelse)
            if tt != et:
                unified:object = promote(tt, et)
                return f"({cond} ? ({unified})({then_code}) : ({unified})({else_code}))", unified
            return f"({cond} ? {then_code} : {else_code})", tt
        if isinstance(node, ast.Call):
            return self._visit_call(node)
        if isinstance(node, ast.Subscript):
            return self._visit_subscript(node)
        if isinstance(node, ast.Attribute):
            return self._visit_attribute(node)
        self._error(f"unsupported expression: {node.__class__.__name__}", node)


    def _visit_enum_constant(self, entry:Tuple[Any, Any], node:ast.AST)->Translated:
        """Emit an enum member as the constant it stands for.

        An `IntEnum` member's value is an int and an `Enum`'s may be anything; only the
        numeric ones have a device meaning, so a member whose value is not a number is
        refused with the value's type named -- there is nothing to emit for it.
        """
        payload:object = entry[0]
        member:object = entry[1]
        name:str = f"{type(member).__name__}.{getattr(member, 'name', '?')}"
        if isinstance(payload, enum.Enum):
            payload = payload.value
        if isinstance(payload, bool):
            return ("true" if payload else "false"), "bool"
        if isinstance(payload, int):
            if payload == 0:
                return "0", "int"
            # An enumerator is spelled in the C source, so a negative one keeps its sign.
            return str(payload), "int"
        if isinstance(payload, float):
            return self._visit_constant(ast.Constant(value=payload))
        self._error(
            f"{name} is an enum member whose value is a {type(payload).__name__}, which has "
            f"no device equivalent: give the member a numeric value (an int or a float), or "
            f"read the value on the host and pass it as a scalar argument", node)


    def _visit_constant(self, node:ast.Constant)->Translated:
        value:object = node.value
        if isinstance(value, bool):
            return ("true" if value else "false"), "bool"
        if isinstance(value, int):
            return str(value), "int"
        if isinstance(value, float):
            # A python float literal has no width of its own, so it takes the one the
            # `float` mapping chooses -- the same rule a python complex literal follows
            # (`1 + 2j` is a c128 under `float=np.float64`), and the same rule a python
            # float *argument* follows. Spelling every literal with the `f` suffix cut a
            # 17-digit constant down to float32 inside a double kernel.
            #
            # The text itself comes from `_real_literal`, which is also the one place that
            # knows how to spell an infinity and a NaN: this branch had a second copy of the
            # suffix rule and that copy emitted a bare `inf` (not a C token), so a kernel
            # with an infinity did not build.
            width:str = "double" if self.type_map.get("float", "float") == "double" else "float"
            return self._real_literal(value, width), width
        if isinstance(value, complex):
            # A complex literal is written with its two parts, because C has no complex
            # constant: the header builds the value from the reals. A python complex has no
            # width of its own, so it takes the one the `float` mapping chooses -- the same
            # rule a python float follows -- which is why `1 + 2j` meets a complex64 operand
            # as a complex64 while `@kernel(float=np.float64)` makes it a complex128.
            kind:str = self._python_complex_kind()
            return self._emit_complex_literal(value.imag, value.real, kind), kind
        self._error(f"unsupported constant: {value!r}", node)


    def _visit_name(self, node:ast.Name)->Translated:
        host:Optional[HostValueEntry] = self.host_values.get(node.id)
        if host is not None:
            # The host created this object; on the device it is just the parameter that
            # carries it (see _host_created_assignment).
            return device_name(node.id), host["type"]
        host_scalar:Optional[DeviceType] = self.host_scalars.get(node.id)
        if host_scalar is not None:
            # A host-region statement bound this value; the device reads the parameter
            # that carries it (see _bind_host_statement).
            return device_name(node.id), host_scalar
        outer:Optional[Dict[str, object]] = self.outer_vars.get(node.id)
        if outer is not None:
            # A writable outer variable travels as `T*`, so every use is a dereference; a
            # read-only one travels as `T` and is simply the parameter itself.
            if outer["writable"]:
                return f"(*{device_name(node.id)})", outer["type"]
            return device_name(node.id), outer["type"]
        t:DeviceType = self._lookup(node.id)
        if isinstance(t, tuple) and len(t) == 2 and t[0] == "carried":
            # One location for the whole launch. A subscript rather than (*name) for the
            # same reason an outer scalar uses one: the read/write analysis only recognizes
            # that form as a write, so the host reads the final value back.
            return f"{device_name(t[1])}[0]", "int"
        if t is not None:
            if isinstance(t, tuple) and len(t) == 5 and t[0] == "shared":
                # A per-group buffer is not itself a pointer: it is one slice per group,
                # and only `buf[blockIdx]` picks the caller's out (see
                # _visit_shared_slice).
                self._error(
                    f"'{node.id}' is a local_array_per_group() buffer, which is one "
                    f"slice per work group: index it with the group first, as "
                    f"'{node.id}[blockIdx]'. A buffer this block indexes directly is "
                    f"`local_array(...)`", node)
            if isinstance(t, tuple) and len(t) == 5 and t[0] == BLOCK_ARRAY:
                # A block-local buffer already *is* this block's storage, so its name stands
                # for the declared array itself: the C name is what an atomic passes as its
                # array, and indexing it is an ordinary element access (see
                # _visit_block_array).
                return t[2], t
            if isinstance(t, tuple) and len(t) == 4 and t[0] == "bodyarr":
                # A private array created in the body is indexed like any array; its extents
                # are constants (see _shape_var).
                return device_name(node.id), (PTR, t[3])
            return device_name(node.id), t
        # CUDA's index vectors are values, not calls (`buf[blockIdx][j]`), so the *name*
        # resolves to the vector the backend builds from its own launch terms.
        if node.id in INDEX_VECTOR_BUILTINS:
            table:Dict[str, Tuple[str, str]] = (
                INDEX_VECTOR_OPENCL if self._opencl_syntax else INDEX_VECTOR_BUILTINS)
            vector_type, code = table[node.id]
            return code, vector_type
        value:object = self.resolve_global(node.id)
        enum_value:Optional[Tuple[Any, Any]] = _enum_member_value(value)
        if enum_value is not None:
            # An enum member is a named constant, so the device reads its value. Python has
            # no device enum type to name here -- the number is what every operator and
            # every function that accepts it already wants.
            return self._visit_enum_constant(enum_value, node)
        if isinstance(value, bool):
            return self._visit_constant(ast.Constant(value=value))
        if isinstance(value, int) and not isinstance(value, np.ndarray):
            # A python int read from the enclosing scope is a *value the caller wrote down
            # somewhere else*, and it has no width of its own: it travels as the width this
            # kernel chose for `int`, exactly like a scalar argument, so it gets the same
            # range check. Without it the check existed only on the parameter path, and a
            # module-level constant (`N = 2 ** 31`) was inlined as a bare literal -- the
            # compiler turned it into `-2147483648` with a warning nobody reads, and the
            # device computed with a number the caller never wrote.
            #
            # This is deliberately HERE and not in `_visit_constant`: a literal written
            # *inside* the body keeps its old behaviour (`0xFFFFFFFF` in a hash mix is
            # unsigned arithmetic on purpose, see tests/test_kernel_scalar_calls.py), while
            # a named constant is the same value as passing it in, so the two paths must
            # agree. A literal BinOp is range-checked by `_visit_binop`'s folded-int check.
            int_type:str = self.type_map.get("int", "int")
            _check_integer_fits(value, int_type)
            return str(value), "int"
        if isinstance(value, float) and not isinstance(value, np.ndarray):
            return self._visit_constant(ast.Constant(value=value))
        self._error(self._unknown_name_message(node), node)


    def _unknown_name_message(self, node:ast.Name)->str:
        """Why a name the body plainly assigns cannot be read here.

        The common way to reach this is an assignment that happens inside a branch: the name
        is a local of the function, but C has no declaration for it until the statement that
        assigns it runs, so reading it after the branch fails. Python says the same thing
        differently -- it raises UnboundLocalError, and only when the branch did not run --
        which is why this reads better as a statement about where the first assignment is
        than as "unknown name".
        """
        name:str = node.id
        if name in self._local_names():
            return (f"'{name}' is assigned inside a branch or loop, so it has no value yet "
                    f"at this point: give it one before the branch (`{name} = 0`) so the "
                    f"branch only updates it")
        return (f"unknown name '{name}' (only parameters, local variables and int/float/bool "
                f"global constants are supported)")


    def _real_literal(self, value:float, width:str)->str:
        """A real literal of the given width, spelled so the C compiler reads it as one.

        Three values need a spelling of their own, because `repr` does not give one the C
        compiler accepts:

        * an infinity became the bare word ``inf``, which is not a C token -- the source did
          not compile at all;
        * a NaN became ``nanf``, a *call* into ``<math.h>``, and whether that header is in
          the prelude is the backend's business, so one backend compiled a kernel another
          refused;
        * a value below the float32 range (``1e-40``) became ``1e-40f``, which the compiler
          rounds to zero -- the device then computed 0.0 where python computes 1e-40.

        All three are answered as self-contained expressions built from literals every C
        dialect accepts: ``(1.0f/0.0f)``, ``(0.0f/0.0f)`` and a width-appropriate
        division. No header, no compiler-specific macro, and the value the device sees is
        the value python wrote.
        """
        if math.isinf(value):
            sign:str = "-" if value < 0 else ""
            return f"({sign}1.0{self._real_suffix(width)}/0.0{self._real_suffix(width)})"
        if math.isnan(value):
            suffix:str = self._real_suffix(width)
            return f"(0.0{suffix}/0.0{suffix})"
        text:str = repr(float(value))
        if not ("." in text or "e" in text or "E" in text):
            text += ".0"
        if width != "double" and not text.endswith(("f", "F", "y")):
            text += "f"
        return text


    @staticmethod
    def _real_suffix(width:str)->str:
        """The literal suffix a real of this width needs (none for a double)."""
        return "" if width == "double" else "f"


    def _condition_code(self, node:ast.expr)->str:
        """A condition as C reads it: `(code)` for what is already a scalar test.

        A complex value is truthy in python when either part is non-zero, so a condition
        holding one (`if z:`, `while z and n < max:`) is rewritten as the comparison the
        record itself cannot express -- the plain `if (z)` reached the compiler and was
        refused with "expression must have bool type". A zero complex value means
        `z.re == 0 && z.im == 0`, which is the negation of the header's own inequality
        against the zero the constructor builds.
        """
        code, t = self._visit_expression(node)
        dtype:Optional[np.dtype] = self._complex_dtype_of(t)
        if dtype is None:
            return f"({code})"
        self._require_complex(dtype)
        kind:str = self._complex_side(t, dtype)
        return (f"({kind}_equal({code}, {kind}_from_real(0)) == 0)")


    def _truth_of(self, code:str, t:object)->str:
        """A translated scalar as a **bool** expression, for a conditional's condition.

        `_condition_code` is right for an `if`/`while` on every backend, but a conditional
        *expression* (`?:`) is stricter: OpenCL C refuses a floating-point condition
        outright ("used type 'float' where floating point type is not allowed") while CUDA
        accepts it, so the same source compiled on one backend and failed on the other. A
        comparison against zero is a bool everywhere and is exactly python's truthiness for
        a number.

        A value that is already a bool needs no comparison -- `(x != 0)` on a bool is legal
        but noisy, and this keeps the emitted text readable.
        """
        dtype:Optional[np.dtype] = self._complex_dtype_of(t)
        if dtype is not None:
            self._require_complex(dtype)
            kind:str = self._complex_side(t, dtype)
            return f"({kind}_equal({code}, {kind}_from_real(0)) == 0)"
        if t == "bool":
            return f"({code})"
        return f"(({code}) != 0)"
