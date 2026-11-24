from __future__ import annotations
import re
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
from .scalar_recursion import (
    mark_in_progress,
)


class _DeviceCallMixin:

    def _emit_device_call(self, name:str, target:object, args:List[Translated], node:ast.Call,
                          self_param:str="",
                          receiver_as_argument:bool=False,
                          tuple_targets:Optional[List[str]]=None,
                          class_constants:Optional[Dict[str, object]]=None,
                          class_owner:Optional[type]=None,
                          leading_skips:int=0)->Translated:
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
        # How many of the callee's own parameters the call does not spell: a method's
        # receiver is inserted ahead of the arguments unless the call passes it as one.
        receiver_offset:int = 1 if (self_param and not receiver_as_argument) else 0
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
                # OpenCL C spells the literal with designated initialisers --
                # `((T){...})`, one more wrapping pair than CUDA's `(T{...})` --
                # so one regex accepts both shapes; without it an OpenCL call
                # reaches `&((T){...})` and the compiler refuses it ("taking the
                # address of a temporary").
                literal_head:re.Pattern[str] = re.compile(
                    rf"^\(\(?{re.escape(value_t)}\)?{{")
                if literal_head.match(code):
                    temporary:str = f"_arg{self._next_temp()}"
                    self._emit(f"{value_t} {temporary} = {code};")
                    code = temporary
                args[index] = (f"&({code})", (PTR, value_t))
        call_codes:List[str] = [c for c, _ in args]
        if self_param and not class_constants:
            # The receiver is a method's first parameter, exactly as in the signature the
            # callee builds (see FunctionTranslator.__init__), so it is passed first. The
            # array attributes come after the caller's arguments, as their own parameters.
            # A cross-class classmethod call carries no receiver at all: the callee's
            # `cls.<attr>` reads are compile-time constants (see class_constants), so the
            # callee's signature has no cls parameter to fill.
            if self.self_struct_members and not receiver_as_argument:
                call_codes.insert(0, self.self_param)
            call_codes.extend(f"_self_{attr}" for attr in self_types)
        # The caller's own arguments come first, then the receiver struct, then the array
        # attributes -- the same order the callee binds them in, and the same order the
        # call codes below follow.
        param_types:List[object] = ([t for _, t in args]
                                 + ([self.self_struct_type]
                                    if self_param and self.self_struct_members
                                    and not receiver_as_argument and not class_constants
                                    else [])
                                 + list(self_types.values()))

        # The helper's generated code embeds where its raise sites land in the *caller's*
        # table (`EXCEPTION_BASE`), so two call sites with different bases are two different
        # functions. Only a raising helper has a base at all, and a non-raising one keys
        # exactly as it always did.
        exception_base:int = len(self.raise_sites)
        sig_key:Tuple[Callable, Tuple[object, ...], int, int] = (
            raw_func, tuple(param_types), exception_base, leading_skips)
        entry:Optional[Dict[str, object]] = self.device_cache.get(sig_key)
        if entry is not None:
            if entry["status"] == "in_progress":
                # The site calls a function that is being translated right now: a
                # cycle. The key alone cannot say which frame -- a self-call's key
                # carries the caller's exception base, which differs from the callee's
                # own whenever an outer caller raises -- so the frame on the
                # in-progress entry (see scalar_recursion) is the authority. The
                # backends that allow recursion and the shapes whose signature the
                # call site can spell completely get native recursion instead of the
                # refusal: the generated C is a plain device function calling itself
                # by name, the code a hand-written CUDA file would carry. Everything
                # else keeps the refusal -- MSL and OpenCL C forbid recursion outright.
                admitted, return_type = self._native_recursion_admitted(raw_func,
                                                                        param_types)
                if admitted:
                    # The generated unit now recurses natively: the launch has to run
                    # with a per-thread stack large enough for the depth the user's
                    # data implies (see uses_native_recursion and the stack raise in
                    # _compile_kernels).
                    self.uses_native_recursion = True
                    return self._emit_recursive_site(raw_func, param_types,
                                                     call_codes, node, name,
                                                     return_type)
                self._error(
                    f"recursive call detected: '{name}' -- this backend's device "
                    f"functions cannot recurse (MSL and OpenCL C forbid it), so the "
                    f"call needs one of: a tail-recursive spelling, which the "
                    f"translator compiles into a loop automatically; a loop written "
                    f"by hand, which every backend runs; or launching this kernel on "
                    f"cuda or hip, whose device functions may recurse natively", node)
            # A cache hit on a cycle partner's *completed* entry needs no special
            # handling: the entry carries the final signature, and the forwarding
            # below is the ordinary one.
            # The callee's outer-scope values must be parameters of THIS caller too, and
            # a cache hit used to skip that registration -- only the miss path required
            # them. A helper that reaches a module constant only through a cached callee
            # then forwarded an argument it never declared: the generated unit referenced
            # `H_R` (or whatever the constant is) and the device compiler refused it with
            # `identifier "H_R" is undefined`. Requiring here keeps every caller of a
            # cached entry on the same footing as the first one.
            stored_outer:object = entry.get("outer_vars") or {}
            if isinstance(stored_outer, dict):
                for outer_name, outer in stored_outer.items():
                    self._require_outer_parameter(outer_name, outer, node)
            stored_params:object = entry.get("shape_params") or ()
            site_params:List[Tuple[str, int]] = (
                [(shape[0], shape[1]) for shape in stored_params]
                if isinstance(stored_params, list) else [])
            argument_codes:List[str] = (
                call_codes
                + self._site_shape_codes(raw_func, site_params, node, receiver_offset)
                + list(entry.get("outer_codes") or []))
            if entry.get("needs_prologue"):
                # The callee reads a work-item index, so it declared the six prologue
                # parameters; the caller passes its own (it has them, being a kernel or
                # another helper that received them the same way).
                argument_codes.extend(self._prologue_parameter_arguments())
            if entry.get("needs_exception"):
                # The callee can raise, so it declared the exception slot and stop flag; the
                # caller forwards its own. The site ids the callee emitted are already
                # absolute (its `raise_site_base` was this caller's total at render time), so
                # adopting its table is a plain extend.
                self._adopt_raise_sites(entry)
                argument_codes.append(self.EXCEPTION_SLOT)
                argument_codes.append(self.EXCEPTION_STOP)
            return (f"{entry['c_name']}({', '.join(argument_codes)})",
                    entry["return_type"])

        mark_in_progress(self.device_cache, sig_key, raw_func, param_types, self)
        try:
            # Through the facade, lazily: this mixin is imported by the module that
            # defines the class, and a top-level import would be a cycle.
            from . import FunctionTranslator
            callee:FunctionTranslator = FunctionTranslator(
                raw_func, param_types, self.backend, _helper_globals(raw_func, self.resolve_global),
                self.device_cache,
                is_kernel=False, self_param=self_param, self_attr_types=self_types,
                self_owner=(class_owner if class_owner is not None else self.self_owner),
                launch_backend=self.launch_backend,
                self_struct_members=(self.self_struct_members
                                     if self_param and not receiver_as_argument
                                     else None),
                self_struct_type=(self.self_struct_type
                                  if self_param and not receiver_as_argument else None),
                receiver_as_argument=receiver_as_argument,
                # A cross-class classmethod call binds the callee's `cls` to a class
                # no caller value represents: its `cls.<attr>` reads are the class
                # attributes' own values, which travel as compile-time constants
                # (see class_constant_values). A same-class call passes the caller's
                # receiver instead, and the constants stay empty.
                class_constants=class_constants or {},
                # A device helper is translated *inside* the kernel's own translation unit, so
                # it has to make the same width choices the kernel does: a python complex
                # literal in a helper body follows the same `float=` mapping (see
                # `_python_complex_kind`). Without this the helper silently answered the
                # default while its caller was mapped to a double.
                type_map=self.type_map,
                leading_skips=leading_skips)
            # A helper that raises writes `base + local id` into the slot, so it has to know
            # where its sites land in *this* caller's table **before** it translates -- the
            # number is baked into the emitted code, which is also why the base is part of
            # the cache key above.
            callee.raise_site_base = exception_base
            callee.translate()
            # The callee's own spelling of its receiver (its first parameter's name):
            # its member-write verdict is read under that name below.
            callee_param_name:str = callee.self_param or self_param
            c_name:str = self._device_func_name(raw_func, param_types)
            # A helper that indexes a multi-dimensional array needs that array's extents, and
            # they are parameters, not part of the pointer. They are appended to the helper's
            # signature under the helper's own names (`render_device` below), and each call
            # site passes the extents of the arrays *it* bound -- spelled in the caller's own
            # terms (`_self_grid_shape1` for an attribute the helper received), because the
            # launch reads the caller's `shapes_used` against the caller's parameter list.
            # The callee-side names travel in the entry (`shape_params`), so a second call
            # site with the same signature binds them to its own arrays instead of reusing
            # the first site's variables.
            shape_params:List[Tuple[str, int]] = list(callee.shapes_used)
            # A helper reaches outer-scope names the same way the kernel does, so it gets
            # the same parameters: the call site forwards its own (see below). The
            # callee's own entries travel with the cache entry, so a *second* caller that
            # hits this entry can require them on itself the way this caller did -- the
            # forwarding codes are the callee's names, and a caller that never declared
            # the parameters would emit a call to names it does not carry.
            outer_codes:List[str] = []
            for outer_name, outer in callee.outer_vars.items():
                self._require_outer_parameter(outer_name, outer, node)
                outer_codes.append(self._outer_argument(outer_name, outer))
            # The cycle this entry closes: the tokens the body's walk left in *this*
            # translator's lines are rewritten before the render -- the code string
            # below is built from these very lines, so the tokens must be gone by
            # then. A self-call's shapes are final here; a mutual partner's call is
            # pure-scalar by the bar the site checked, so its call needs no shape
            # argument (see _finish_recursive_sites).
            callee._finish_recursive_sites()
            # A callee that writes its receiver's members makes THIS caller's receiver
            # writable too: the same structure travels down, the stores land in it, and
            # the launch has to read it back. The caller's signature must lose its
            # `const` (the callee's own verdict is what decided the callee's), or C
            # rejects the pointer hand-off and the device write never comes home. The
            # same forwarding a `const_cast` at the call site states for C states here
            # for the analysis: member writes flow out of the callee onto the caller.
            if (self_param and not receiver_as_argument and self.self_struct_members
                    and getattr(callee, "member_writes", None)
                    and callee_param_name in callee.member_writes):
                self.member_writes.add(self.self_param)
            entry = {
                "status": "done",
                "c_name": c_name,
                "return_type": callee.return_type,
                # A tuple return's element types: the call site appends one `T*`
                # destination per element and passes it after every other argument
                # (render_device appends the matching declarations).
                "tuple_return_types": list(callee.tuple_return_types or []),
                "needs_prologue": callee._needs_prologue_parameters(),
                # The callee can raise, so its signature carries the exception slot and stop
                # flag; the call site forwards the caller's own (render_device appends the
                # matching declarations after everything else).
                "needs_exception": bool(getattr(callee, "has_raises", False)),
                # The raise sites this helper owns, with ids local to it. The caller rebases
                # them onto its own list when it forwards the call (see the call-site
                # branches), because the id the device writes indexes the *kernel's* table.
                "raise_sites": list(getattr(callee, "raise_sites", None) or []),
                "code": callee.render_device(c_name, shape_params),
                "shape_params": [list(shape) for shape in shape_params],
                "outer_codes": outer_codes,
                # The callee's outer-scope entries, in the order `outer_codes` forwards
                # them. A hit on this entry must require them on the *hitting* caller
                # exactly as the miss path did (see the hit branch below).
                "outer_vars": dict(callee.outer_vars),
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
            # Recorded after `render_device`, so a definition the render itself
            # contributed (see _render_body's declaration fixup) is carried too: the
            # write step emits these beside the entry's own code, in whichever file
            # that code lands (see _compile_write's own and foreign prefixes).
            entry["device_helpers"] = list(callee.device_helpers)
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

        argument_codes:List[str] = (
            call_codes
            + self._site_shape_codes(raw_func, shape_params, node, receiver_offset)
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
        if entry.get("needs_exception"):
            # The callee can raise, so it declared the exception slot and stop flag; the
            # caller forwards its own. Its ids are already absolute (see the cache-key note).
            self._adopt_raise_sites(entry)
            argument_codes.append(self.EXCEPTION_SLOT)
            argument_codes.append(self.EXCEPTION_STOP)
        return (f"{entry['c_name']}({', '.join(argument_codes)})", entry["return_type"])


    def _adopt_raise_sites(self, entry:Dict[str, object])->None:
        """Take a raising helper's sites onto this translation unit's own table.

        The helper emitted `raise_site_base + local id` into its code, and the base was this
        caller's total when the helper was rendered -- so the ids already line up with the
        positions the sites take here, and adopting them is a plain append.
        """
        if not entry.get("needs_exception"):
            return
        self.has_raises = True
        sites:object = entry.get("raise_sites") or []
        if not isinstance(sites, list):
            return
        for site in sites:
            self.raise_sites.append(dict(site))  # type: ignore[arg-type]


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


    def _site_shape_codes(self, raw_func:Callable, shape_params:List[Tuple[str, int]],
                          node:ast.Call, receiver_offset:int)->List[str]:
        """The shape-variable arguments one call site passes for a helper's extents.

        The helper's signature declares each extent under the helper's own name
        (`const int _grid_shape1`, see `render_device`); the argument this site passes
        is the extent of the array *this site* bound to that parameter, spelled in the
        caller's own terms (`_self_grid_shape1` when the array is the receiver's `grid`
        attribute). The registration rides on `_shape_var`/`_tail_var`, so the caller's
        `shapes_used` gains the caller-side name -- the one the launch can resolve
        against the caller's parameter list -- and the generated call hands the helper
        the caller's variable.
        """
        codes:List[str] = []
        for array_name, dim in shape_params:
            bound:str = self._bound_shape_name(raw_func, array_name, node, receiver_offset)
            codes.append(self._shape_var(bound, dim) if dim >= 0
                         else self._tail_var(bound, -dim - 2))
        return codes


    def _bound_shape_name(self, raw_func:Callable, array_name:str, node:ast.Call,
                          receiver_offset:int)->str:
        """The caller-side spelling of the array one helper shape extent names.

        The callee's `shapes_used` names are the callee's own: a parameter it was
        handed (`grid`), its own receiver attribute (`_self_grid`), or an outer name.
        The last two are already caller terms -- the attributes are forwarded under
        the caller's own names and an outer variable travels by name -- so only a
        *parameter* needs the call's binding: the argument this call passed in that
        parameter's position names the array on the caller's side. A bare name is
        rooted through the pointer-alias chain (`src = self.grid` makes `src` read as
        `_self_grid`), and a `self.grid` argument names the attribute slot directly.
        An argument this mapping cannot read -- an expression, a slice, a call's
        result -- leaves the callee's own name, which is what the shape variable was
        called before this mapping existed.
        """
        code:object = getattr(raw_func, "__code__", None)
        if code is not None:
            params:Tuple[str, ...] = code.co_varnames[:code.co_argcount]
            if array_name in params:
                position:int = params.index(array_name) - receiver_offset
                if 0 <= position < len(node.args):
                    argument:ast.expr = node.args[position]
                    if isinstance(argument, ast.Name):
                        return self._array_pointer_root(argument.id)
                    if (self.self_param and isinstance(argument, ast.Attribute)
                            and isinstance(argument.value, ast.Name)
                            and argument.value.id == self.self_param):
                        return f"_self_{argument.attr}"
        return array_name


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
