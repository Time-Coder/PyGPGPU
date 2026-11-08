# A kernel launched from a kernel's device code: rendered ahead of its caller.
from __future__ import annotations
import ast
import hashlib
from typing import (Callable, List, Optional)
from .notation import (
    Translated,
)
from .receivers import (
    _helper_globals,
)
from .predicates import (
    is_int,
)


class _NestedLaunchMixin:

    def _emit_nested_launch(self, node:ast.Call)->bool:
        """`inner[(global, local)](args)` -> a device-side launch; True when this was one."""
        func:ast.expr = node.func
        if not isinstance(func, ast.Subscript):
            return False
        # Peel every bracket: child[(n,)] and the invalid child["cuda"][(n,)] both reach
        # a Name at the bottom, and the second has to be named as the mistake it is.
        callee:ast.expr = func.value
        while isinstance(callee, ast.Subscript):
            callee = callee.value
        if not isinstance(callee, ast.Name):
            return False
        target:object = self.resolve_global(callee.id)
        if not getattr(target, "is_pykernel", False):
            return False
        self._render_nested_launch(callee.id, target, func.slice, node)
        return True


    def _render_nested_launch(self, name:str, target:object, work:ast.expr,
                              node:ast.Call)->None:
        """Emit the device-side launch statement for one nested call."""
        if self.launch_backend != "cuda":
            # CUDA calls this dynamic parallelism; the OpenCL kernel language has no
            # nested-launch statement at all, and HIP does not offer it on this platform.
            self._error(
                f"'{name}' is a @kernel launched from device code, which {self.launch_backend} "
                f"does not support: device-side launch needs CUDA's dynamic parallelism. Run "
                f"this kernel on cuda, or call a plain device helper instead", node)
        entries:List[ast.expr] = []
        probe:ast.expr = node.func
        while isinstance(probe, ast.Subscript):
            entries.append(probe.slice)
            probe = probe.value
        for entry in entries:
            if isinstance(entry, ast.Constant) and isinstance(entry.value, str):
                self._error(
                    f"'{name}' is launched from device code, where the backend and the device "
                    f"cannot be chosen: the device is already running. Give only the work "
                    f"size, as `{name}[(global, local)](...)`", node)

        if isinstance(work, ast.Tuple):
            if len(work.elts) > 2:
                self._error(
                    f"a device-side launch takes at most (global, local): '{name}' was given "
                    f"{len(work.elts)} work sizes", work)
            parts:List[Translated] = [self._visit_expression(e) for e in work.elts]
            global_code, global_type = parts[0]
            local_code, local_type = parts[1] if len(parts) > 1 else ("", None)
        else:
            global_code, global_type = self._visit_expression(work)
            local_code, local_type = "", None
        if not is_int(global_type):
            self._error(f"the work size of '{name}' must be an integer", work)
        if local_code and not is_int(local_type):
            self._error(f"the local work size of '{name}' must be an integer", work)

        call_args:str = ", ".join(self._visit_expression(a)[0] for a in node.args)
        c_name:str = self._translate_nested_kernel(name, target, node)
        block:str = local_code if local_code else "1"
        grid:str = f"(({global_code}) / ({block}))"
        self._emit(f"{c_name}<<<{grid}, {block}>>>({call_args});")


    def _translate_nested_kernel(self, name:str, target:object, node:ast.Call)->str:
        """Render the launched kernel as another `__global__` function of this source."""
        raw_func:Callable = target.raw_func
        known:Optional[str] = self.nested_seen.get(raw_func)
        if known is not None:
            return known
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
        self.nested_kernels.append(callee.render_kernel(c_name))
        for deeper in callee.nested_kernels:
            if deeper not in self.nested_kernels:
                self.nested_kernels.append(deeper)
        from ..decorator import _dependency_order

        for entry in _dependency_order(self.device_cache):
            code:str = entry["code"]
            if code not in self.nested_helpers:
                self.nested_helpers.append(code)
        return c_name
