from __future__ import annotations
import ast
import os
import types
from typing import (Callable, Dict, List, Optional, Tuple)
from ..structs import dtype_for as structs_dtype_for
from ..device_builtins import (
    python_floor_name,
    python_floor_source,
)
from .notation import (
    PTR,
    Translated,
)
from .receivers import (
    _helper_globals,
)
from .vectors import (
    is_vec,
)
from .walk import (
    short_hash,
    type_sig,
)


class _DeviceCallMixin:

    def _emit_device_call(self, name:str, target:object, args:List[Translated], node:ast.Call,
                          self_param:str="",
                          receiver_as_argument:bool=False,
                          tuple_targets:Optional[List[str]]=None)->Translated:
        is_kernel_ref:bool = getattr(target, "is_pykernel", False)
        raw_func:Callable = (target.raw_func if is_kernel_ref else target)
        if is_kernel_ref:
            # A @kernel with a work size is a nested launch, handled in _visit_call before
            # this point; reaching here means it was called as if it were a helper.
            self._error(
                f"'{name}' is a @kernel function: from device code it is a *launch*, so give "
                f"it a work size -- `{name}[(total, local)](...)` -- rather than calling it "
                f"as a helper", node)
        if not isinstance(raw_func, types.FunctionType):
            # A builtin or a bound builtin method has no python source to translate, so it
            # cannot become a device function. Say which name it was rather than failing
            # inside inspect.getsource.
            self._error(
                f"'{name}' is not a python function this translator can compile into device "
                f"code (only functions defined in python source are supported)", node)

        # A helper that is a method of the kernel's own class keeps the same receiver:
        # its self.xxx references become extra _self_xxx parameters appended after the
        # caller's, and the call site passes them through in that order.
        #
        # A *value* method (`ray.at(t)`) is different: the receiver is already the call's
        # first argument, so it binds as the callee's own first positional parameter
        # (`def at(self, t)` is `(Ray* self, float t)`) and declares no `_self_` slot.
        self_types:Dict[str, object] = ({} if receiver_as_argument
                                     else self.self_attr_types if self_param else {})
        # A structure argument always travels as a pointer, so a local value is passed by
        # its address: the callee reads the members through `->` and a store lands in the
        # caller's own value, which is what a method's receiver needs.
        for index, (code, value_t) in enumerate(args):
            if (isinstance(value_t, str) and structs_dtype_for(value_t) is not None
                    and not is_vec(value_t)):
                # A compound literal has no address: `&((T{...}))` is not valid C, so a
                # value built straight into the call (`self.hit(ray, Interval(0, inf))`)
                # is stored in a local first. The temporary is emitted before this
                # statement, which is the same shape a named construction already takes.
                if code.startswith(f"({value_t}{{"):
                    temporary:str = f"_arg{self._next_temp()}"
                    self._emit(f"{value_t} {temporary} = {code};")
                    code = temporary
                args[index] = (f"&({code})", (PTR, value_t))
        call_codes:List[str] = [c for c, _ in args]
        if self_param:
            # The receiver is a method's first parameter, exactly as in the signature the
            # callee builds (see FunctionTranslator.__init__), so it is passed first. The
            # array attributes come after the caller's arguments, as their own parameters.
            if self.self_struct_members and not receiver_as_argument:
                call_codes.insert(0, self.self_param)
            call_codes.extend(f"_self_{attr}" for attr in self_types)
        # The caller's own arguments come first, then the receiver struct, then the array
        # attributes -- the same order the callee binds them in, and the same order the
        # call codes below follow.
        param_types:List[object] = ([t for _, t in args]
                                 + ([self.self_struct_type]
                                    if self_param and self.self_struct_members
                                    and not receiver_as_argument else [])
                                 + list(self_types.values()))

        sig_key:Tuple[Callable, Tuple[object, ...]] = (raw_func, tuple(param_types))
        entry:Optional[Dict[str, object]] = self.device_cache.get(sig_key)
        if entry is not None:
            if entry["status"] == "in_progress":
                self._error(f"recursive call detected: '{name}'", node)
            argument_codes:List[str] = (call_codes + list(entry.get("shape_codes") or [])
                                        + list(entry.get("outer_codes") or []))
            if entry.get("needs_prologue"):
                # The callee reads a work-item index, so it declared the six prologue
                # parameters; the caller passes its own (it has them, being a kernel or
                # another helper that received them the same way).
                argument_codes.extend(self._prologue_parameter_arguments())
            return (f"{entry['c_name']}({', '.join(argument_codes)})",
                    entry["return_type"])

        self.device_cache[sig_key] = {"status": "in_progress"}
        try:
            # Through the facade, lazily: this mixin is imported by the module that
            # defines the class, and a top-level import would be a cycle.
            from . import FunctionTranslator
            callee:FunctionTranslator = FunctionTranslator(
                raw_func, param_types, self.backend, _helper_globals(raw_func, self.resolve_global),
                self.device_cache,
                is_kernel=False, self_param=self_param, self_attr_types=self_types,
                self_owner=self.self_owner, launch_backend=self.launch_backend,
                self_struct_members=(self.self_struct_members
                                     if self_param and not receiver_as_argument
                                     else None),
                self_struct_type=(self.self_struct_type
                                  if self_param and not receiver_as_argument else None),
                receiver_as_argument=receiver_as_argument,
                # A device helper is translated *inside* the kernel's own translation unit, so
                # it has to make the same width choices the kernel does: a python complex
                # literal in a helper body follows the same `float=` mapping (see
                # `_python_complex_kind`). Without this the helper silently answered the
                # default while its caller was mapped to a double.
                type_map=self.type_map)
            callee.translate()
            c_name:str = self._device_func_name(raw_func, param_types)
            # A helper that indexes a multi-dimensional array needs that array's extents, and
            # they are parameters, not part of the pointer. They are appended to the helper's
            # signature and passed from the caller's own_ shape variables, which the kernel
            # already receives.
            shape_params:List[Tuple[str, int]] = list(callee.shapes_used)
            for array_name, dim in shape_params:
                if (array_name, dim) not in self.shapes_used:
                    self.shapes_used.append((array_name, dim))
            shape_codes:List[str] = [self._shape_var(array_name, dim) if dim >= 0
                                     else self._tail_var(array_name, -dim - 2)
                                     for array_name, dim in shape_params]
            # A helper reaches outer-scope names the same way the kernel does, so it gets
            # the same parameters: the call site forwards its own (see below).
            outer_codes:List[str] = []
            for outer_name, outer in callee.outer_vars.items():
                self._require_outer_parameter(outer_name, outer, node)
                outer_codes.append(self._outer_argument(outer_name, outer))
            entry = {
                "status": "done",
                "c_name": c_name,
                "return_type": callee.return_type,
                # A tuple return's element types: the call site appends one `T*`
                # destination per element and passes it after every other argument
                # (render_device appends the matching declarations).
                "tuple_return_types": list(callee.tuple_return_types or []),
                "needs_prologue": callee._needs_prologue_parameters(),
                "code": callee.render_device(c_name, shape_params),
                "shape_codes": shape_codes,
                "outer_codes": outer_codes,
                # Structure types this helper's signature or body mentions: the caller emits
                # their typedefs once, ahead of every device function.
                "structs": list(callee.private_structs),
                # The python file this function was written in: its device code goes to
                # that module's own source file (see pygpgpu.kernel.source_paths).
                "source_file": getattr(getattr(raw_func, "__code__", None),
                                        "co_filename", "") or "",
                # A helper's body may use quaternion arithmetic too; its generated functions
                # travel with the entry so the compile step can emit them once.
                "quaternion_helpers": dict(callee.quaternion_helpers),
                "quaternion_needs": set(callee.quaternion_needs),
                # The same for complex values: a helper's body may use them too.
                "complex_needs": set(callee.complex_needs),
            }
            self.device_cache[sig_key] = entry
        except Exception:
            self.device_cache.pop(sig_key, None)
            raise

        # Device code has no shared state or atomics-free write-back for images, so an
        # image a helper samples or stores into has to be declared by the kernel itself.
        if callee.required_image_samplers or callee.required_image_surfaces:
            self._error(
                f"'{name}' uses an image parameter from a device helper, which is not "
                f"supported: sample or store the image in the kernel body instead", node)

        argument_codes:List[str] = (call_codes + list(entry.get("shape_codes") or [])
                                    + list(entry.get("outer_codes") or []))
        if entry.get("needs_prologue"):
            # The callee declared Metal's six prologue parameters because its body reads
            # a work-item index; the caller has them (it is a kernel, or a helper that
            # received them the same way) and forwards them here.
            argument_codes.extend(self._prologue_parameter_arguments())
        if tuple_targets is not None:
            # A tuple return: the destinations the call site unpacks into, appended in
            # the helper's own order (see render_device, which declares them last).
            argument_codes.extend(tuple_targets)
        return (f"{entry['c_name']}({', '.join(argument_codes)})", entry["return_type"])


    def _device_func_name(self, func:Callable, types:List[object])->str:
        # A method of a class reads `Class_method`; a plain function keeps its own name.
        # The hash stays in the name because a helper monomorphizes per argument type
        # (`blend` over float and over float3 are two device functions), and two modules
        # can define the same helper name -- the hash covers both. Nothing is appended to
        # settle a collision: the hash is what keeps two helpers apart, so a numbered
        # suffix would only appear if the hash itself collided.
        base:str = func.__name__
        # A method is named after the class it is *defined on*, which `__qualname__` records:
        # `Ray.at` is `Ray_at`, never the receiving kernel's own class. A plain function has
        # no dot in its qualname, and a nested definition (`<locals>` in it) has no class.
        qualname:str = getattr(func, "__qualname__", "")
        if "." in qualname:
            owner_name:str = qualname.rsplit(".", 1)[0]
            if "<" not in owner_name:
                base = f"{owner_name}_{func.__name__}"
        module_file:str = getattr(getattr(func, "__code__", None), "co_filename", "")
        module_mark:str = os.path.basename(module_file) if module_file else ""
        signature:str = module_mark + '|' + type_sig(types)
        # The `int=`/`float=` mapping belongs in the hash for the same reason it is part of the
        # kernel's own C name: a python literal's width follows it, so one helper translated
        # under two mappings is two device functions -- with different bodies, and for a
        # complex value even a different return type. A source file carries every kernel that
        # was written in it, so without the mapping the two definitions would land on one name
        # and a C++ backend rejects them as "cannot overload functions distinguished by return
        # type alone".
        mapping_key:str = ",".join(f"{key}={self.type_map[key]}"
                                   for key in sorted(self.type_map))
        if mapping_key:
            signature += '|' + mapping_key
        return f"{base}_{short_hash(signature)}"


    def _device_qualifier(self)->str:
        """The storage qualifier a device function needs on this backend."""
        return "__device__ " if self.backend == "cuda" else ""


    def _emit_floored(self, operations:Tuple[str, str], operation:str,
                      left:str, ctype:str, right:str, right_t:object,
                      node:ast.AST)->Translated:
        """`left // right` or `left % right` with python's floor semantics, for integers.

        C's `/` truncates toward zero and its `%` takes the dividend's sign; python floors
        and takes the divisor's sign. They agree only when the remainder is zero or the
        operands have the same sign, so the generated code calls a small device function
        that applies the one-step correction (see `python_floor_source`).

        An **unsigned** width needs no correction -- its remainder is never negative -- and
        `(r < 0)` on it is always false anyway, which some compilers warn about, so the
        plain C operator is emitted there and the helper is not declared.

        The helper is declared once per (operation, width): a file that uses both `//` and
        `%` on `int` carries two functions, and a second kernel using `//` on `int` finds
        the definition already recorded.
        """
        device_ctype:str = self._device_type(ctype)
        helper_name:str = python_floor_name(operation, device_ctype)
        source:str = python_floor_source(operation, device_ctype, self.backend)
        if not source:
            # An unsigned width: the C operator already matches python.
            symbol:str = "/" if operation == "floor_div" else "%"
            return f"({left} {symbol} {right})", ctype
        if source not in self.device_helpers:
            self.device_helpers.append(source)
        return f"{helper_name}({left}, {right})", ctype
