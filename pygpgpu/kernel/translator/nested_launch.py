# A kernel launched from a kernel's device code: rendered ahead of its caller.
from __future__ import annotations
import ast
import hashlib
import warnings
from typing import (Callable, Dict, List, Optional, Tuple)
from .notation import (
    Translated,
)
from .receivers import (
    _helper_globals,
)
from .predicates import (
    is_int,
)


class NestedLaunchWarning(Warning):
    """A device-side launch carried something that has no meaning there, and was ignored."""


def _bracket_entries(func:ast.expr)->List[ast.expr]:
    """Every bracket the callable was given, outermost first.

    `child["cuda"][(n,)](...)` subscripts twice, and one of those entries is a device
    selector rather than a work size -- which is exactly the mistake worth reporting, so the
    entries are collected here and classified by the caller instead of taking `.slice` of the
    first one only.
    """
    entries:List[ast.expr] = []
    probe:ast.expr = func
    while isinstance(probe, ast.Subscript):
        entries.append(probe.slice)
        probe = probe.value
    return entries


def _is_work_size(entry:ast.expr)->bool:
    """True for a bracket entry that states a work size rather than naming a device."""
    return not (isinstance(entry, ast.Constant) and isinstance(entry.value, str))


class _NestedLaunchMixin:

    def _nested_extents(self)->Dict[Callable, List[str]]:
        """callee -> the call-site expressions for the extent parameters it derives.

        Created on first use rather than in `__init__`, because only this mixin reads it: the
        rest of the translator never needs to know that a nested launch forwards extents.
        """
        extents:Optional[Dict[Callable, List[str]]] = getattr(self, "_nested_extent_cache", None)
        if extents is None:
            extents = {}
            self._nested_extent_cache = extents
        return extents

    def _emit_nested_launch(self, node:ast.Call)->bool:
        """`inner[(global, local)](args)` / `inner(args)` -> a device-side launch.

        True when this call was one. A bare call is a launch too when the child declares its
        own domain with a magic loop: the child's loop names the extent, so the launch
        geometry is derived exactly as a host launch derives it -- the two spellings of one
        kernel must not disagree about whether a work size is needed.
        """
        func:ast.expr = node.func
        # Peel every bracket: `child[(n,)]` and `child["cuda"][(n,)]` both reach a Name at the
        # bottom. The second is not a mistake to refuse -- the device is already running, so
        # the selector is ignored with a warning (see _render_nested_launch).
        callee:ast.expr = func
        while isinstance(callee, ast.Subscript):
            callee = callee.value
        if not isinstance(callee, ast.Name):
            return False
        target:object = self.resolve_global(callee.id)
        if not getattr(target, "is_pykernel", False):
            return False
        if not isinstance(func, ast.Subscript) and not self._child_derives_its_size(
                target, node):
            # A bare call of a kernel that declares no domain has no geometry to launch
            # with, so it is the "called a kernel as a helper" mistake rather than a launch.
            return False
        self._render_nested_launch(callee.id, target, node)
        return True


    def _child_derives_its_size(self, target:object, node:ast.Call)->bool:
        """True when a bare call of `target` can derive its own work size from its body.

        That is the case a host launch with no bracket also handles: the child carries a magic
        loop, so the loop's own array supplies the extent. A gid-style child declares its
        domain outside the body, and a device-side launch has nowhere to read that from -- a
        bare call of one stays the helper mistake.
        """
        from ..decorator import _declares_a_magic_loop

        return bool(_declares_a_magic_loop(target.raw_func))


    def _render_nested_launch(self, name:str, target:object,
                              node:ast.Call)->None:
        """Emit the device-side launch statement for one nested call."""
        if self.launch_backend != "cuda":
            # CUDA calls this dynamic parallelism; the OpenCL kernel language has no
            # nested-launch statement at all, and HIP does not offer it on this platform.
            self._error(
                f"'{name}' is a @kernel launched from device code, which {self.launch_backend} "
                f"does not support: device-side launch needs CUDA's dynamic parallelism. Run "
                f"this kernel on cuda, or call a plain device helper instead", node)
        # A child launch happens on the device that is already running, so a selector in a
        # bracket cannot choose anything: the device is decided, not re-decided. Ignoring it
        # (with a warning naming it) keeps `child["cuda"][(n,)](...)` -- the spelling a
        # reader may carry over from a host launch site -- working from device code, instead
        # of failing on an argument that has no meaning there. The work-size bracket is
        # still at most one, and it is what the launch geometry comes from.
        entries:List[ast.expr] = _bracket_entries(node.func)
        work_entries:List[ast.expr] = [
            entry for entry in entries if _is_work_size(entry)]
        selectors:List[str] = [entry.value for entry in entries
                               if isinstance(entry, ast.Constant)
                               and isinstance(entry.value, str)]
        if selectors:
            warnings.warn(
                f"'{name}' is launched from device code, where the device is already running, "
                f"so the selector {', '.join(repr(selector) for selector in selectors)} has no "
                f"effect on the child and is ignored: a child launch takes only its work size, "
                f"`{name}[(global, local)](...)`",
                NestedLaunchWarning, stacklevel=1)
        if len(work_entries) > 1:
            self._error(
                f"a device-side launch takes at most one work size, (global[, local]): "
                f"'{name}' was given {len(work_entries)}", node)

        call_args:str = ", ".join(self._visit_expression(a)[0] for a in node.args)
        c_name:str
        extents:List[str]
        c_name, extents = self._translate_nested_kernel(name, target, node)
        # The child's signature ends with one scalar per extent it derives from an array
        # (`int _out_shape0`), exactly as a host launch supplies them after the user's own
        # arguments. A device-side launch has to do the same, and it can: the parent already
        # holds the arrays the child is launched over, so the extent is `a.shape[dim]` there.
        if extents:
            call_args = ", ".join([call_args, *extents]) if call_args else ", ".join(extents)

        if not work_entries:
            # No work size was written, so it is derived from the child's own magic loop --
            # the same automatic size a host call gets. A bare call only reaches here when
            # `_child_derives_its_size` said the body declares a domain; a selector-only
            # bracket has no domain at all and is refused here instead of emitting a
            # geometry-less kernel call.
            if selectors and not extents:
                self._error(
                    f"'{name}' is launched from device code and its bracket names a device "
                    f"instead of a work size, and the body declares no magic loop to derive "
                    f"one from. Give the work size: `{name}[(global, local)](...)`. A device "
                    f"selector has no meaning in a child launch -- the device is already "
                    f"running", node)
            global_code = self._derived_child_size(name, node, extents)
            local_code, local_type = "", None
            global_type = "int"
        else:
            work:ast.expr = work_entries[0]
            if isinstance(work, ast.Tuple):
                if len(work.elts) > 2:
                    self._error(
                        f"a device-side launch takes at most (global, local): '{name}' was "
                        f"given {len(work.elts)} work sizes", work)
                parts:List[Translated] = [self._visit_expression(e) for e in work.elts]
                global_code, global_type = parts[0]
                local_code, local_type = parts[1] if len(parts) > 1 else ("", None)
            else:
                global_code, global_type = self._visit_expression(work)
                local_code, local_type = "", None
        if not is_int(global_type):
            self._error(f"the work size of '{name}' must be an integer", node)
        if local_code and not is_int(local_type):
            self._error(f"the local work size of '{name}' must be an integer", node)

        block:str = local_code if local_code else "1"
        grid:str = f"(({global_code}) / ({block}))"
        self._emit(f"{c_name}<<<{grid}, {block}>>>({call_args});")


    def _derived_child_size(self, name:str, node:ast.Call,
                            extents:List[str])->str:
        """The child's work size when the call wrote none: its own magic loop's extent.

        The child's extent parameters are exactly its loop bound (`int _out_shape0`), and the
        launch already forwards them to the child, so the same expressions are the geometry.
        A child with several magic loops is a multi-pass kernel, which a device-side launch
        cannot start as one call, so it is refused rather than silently run as its first pass.
        """
        if not extents:
            self._error(
                f"'{name}' is launched from device code without a work size, and its body "
                f"declares no magic loop to derive one from. Write the size: "
                f"`{name}[(global, local)](...)`", node)
        return f"({extents[0]})"


    def _translate_nested_kernel(self, name:str, target:object,
                                 node:ast.Call)->Tuple[str, List[str]]:
        """Render the launched kernel as another `__global__` function of this source.

        Returns the child's generated name and the call-site expressions for the extent
        parameters its signature derives. Those extents are the child's own work size (a
        magic loop over an argument's shape becomes `int _<arg>_shape<dim>`), so the parent
        has to hand them over -- the same values a host launch appends after the user's
        arguments. They come from the *parent's* view of the same array, which is what makes
        the child's launch geometry agree with the body it translates.
        """
        raw_func:Callable = target.raw_func
        known:Optional[str] = self.nested_seen.get(raw_func)
        if known is not None:
            return known, self._nested_extents().get(raw_func, [])
        param_types:List[object] = [self._visit_expression(a)[1] for a in node.args]
        from ..decorator import _positional_argument_names

        expected:int = len(_positional_argument_names(raw_func))
        if len(param_types) != expected:
            self._error(
                f"'{name}' takes {expected} argument(s) but a device-side launch gave "
                f"{len(param_types)}", node)
        digest:str = hashlib.md5(
            (raw_func.__name__ + repr(param_types)).encode("utf-8")).hexdigest()[:8]
        c_name:str = f"nested{len(self.nested_kernels)}_{digest}"
        # Registered before translating, so a kernel that launches itself is refused rather
        # than recursed into.
        self.nested_seen[raw_func] = c_name
        # Through the facade, lazily: this mixin is imported by the module that
        # defines the class, and a top-level import would be a cycle.
        from . import FunctionTranslator
        callee:FunctionTranslator = FunctionTranslator(
            raw_func, param_types, self.backend, _helper_globals(raw_func, self.resolve_global),
            self.device_cache, is_kernel=True, self_param="", self_attr_types={},
            self_owner=None, launch_backend=self.launch_backend, type_map=self.type_map)
        callee.translate()
        # The extent parameters the child's signature will carry, paired with how the parent
        # names the same array: the child's parameter names ARE the parent's argument names
        # for a positional device-side launch (`child[(n,)](a, out)` binds `a` to `a`).
        extents:List[str] = self._nested_extent_arguments(callee, node)
        self._nested_extents()[raw_func] = extents
        self.nested_kernels.append(callee.render_kernel(c_name))
        for deeper in callee.nested_kernels:
            if deeper not in self.nested_kernels:
                self.nested_kernels.append(deeper)
        from ..decorator import _dependency_order

        for entry in _dependency_order(self.device_cache):
            code:str = entry["code"]
            if code not in self.nested_helpers:
                self.nested_helpers.append(code)
        return c_name, extents


    def _nested_extent_arguments(self, callee:object,
                                 node:ast.Call)->List[str]:
        """The call-site expressions for the child's derived extent parameters.

        A child kernel whose magic loop runs over `out.shape[0]` declares `int _out_shape0`
        after its own parameters. The parent already holds the same array (it is passing it),
        so the argument is that array's extent -- translated by *this* translator, which
        knows how a `.shape` read becomes a device expression (a plain scalar parameter, or
        the sender's own constant). Handing over a raw `out.shape[0]` string would emit a
        member access on a pointer, which the C compiler rejects.

        An extent the parent cannot express is refused by name rather than silently omitted:
        a missing argument is a C compile error far from the line that caused it.
        """
        extents:List[str] = []
        from ..decorator import _positional_argument_names

        bound_names:Dict[str, str] = {}
        for parameter, argument in zip(_positional_argument_names(callee.func), node.args):
            if isinstance(argument, ast.Name):
                bound_names[parameter] = argument.id
        for array_name, dim in callee.shapes_used:
            if dim < 0:
                self._error(
                    f"a device-side launch cannot forward a partial-store extent for "
                    f"'{array_name}'; give the child a magic loop over a whole axis "
                    f"(`for i in range(x.shape[0])`) or launch it from the host", node)
            source_name:Optional[str] = bound_names.get(array_name, array_name)
            shape_read:ast.Subscript = ast.Subscript(
                value=ast.Attribute(value=ast.Name(id=source_name, ctx=ast.Load()),
                                    attr="shape", ctx=ast.Load()),
                slice=ast.Constant(value=dim), ctx=ast.Load())
            ast.copy_location(shape_read, node)
            extents.append(self._visit_expression(shape_read)[0])
        return extents
