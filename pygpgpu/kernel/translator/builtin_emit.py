from __future__ import annotations
import ast
from typing import (Dict, List, Optional, cast)
import numpy as np
from .builtin_python import _BuiltinPythonMixin
from ..builtin_registry import builtin_name_of
from ..device_library import library_function_name, library_functions
from ..device_builtins import (
    ATOMIC_BUILTINS,
    CUDA_ONLY_BUILTINS,
    CUDA_ONLY_CONVERSIONS,
    GENERIC_BUILTINS,
    HIP_ROUND_TOWARD_ZERO_HELPERS,
    HIP_ROUND_TOWARD_ZERO_SOURCES,
    INDEX_BUILTINS,
    SHARED_MATH_BUILTINS,
    SHARED_MATH_TEMPLATES,
    emit_atomic,
    emit_barrier,
    emit_index,
)
from .notation import (
    BLOCK_ARRAY,
    PTR,
    Translated,
    WORK_GROUP_VALUES,
)
from .vectors import (
    _vec_field_names,
    _vec_registry,
    is_vec,
)
from .builtins import (
    COMPLEX_BUILTIN_FUNCTIONS,
    COMPLEX_BUILTIN_REAL_RESULTS,
    COMPLEX_BUILTIN_REFUSALS,
)
from .predicates import (
    is_image,
    is_int,
    is_ptr,
)


class _BuiltinEmitMixin(_BuiltinPythonMixin):

    def _builtin_call_name(self, node:ast.expr)->Optional[str]:
        """The builtin this call invokes, or None when it invokes something else.

        Identity, not spelling: the callee is resolved the way python would resolve it and
        the resulting object is looked up in the registry, so a bare name (``dot(...)``), an
        attribute of the package (``pygpgpu.dot(...)``) and a name imported from the module
        that defines it all agree, while a user function that happens to share a name is
        recognized as theirs.
        """
        if not isinstance(node, ast.Call):
            return None
        callee:ast.expr = node.func
        if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Name):
            owner:object = self.resolve_global(callee.value.id)
            if owner is None:
                return None
            return builtin_name_of(getattr(owner, callee.attr, None))
        if isinstance(callee, ast.Name):
            return builtin_name_of(self.resolve_global(callee.id))
        return None


    def _library_call_name(self, node:ast.expr)->Optional[str]:
        """The registered device function this call invokes, or None.

        The same identity rule as the builtins: the callee is resolved the way python
        would resolve it and the resulting object is looked up in the registry, so a
        user's own function that happens to share a name is not mistaken for the
        registered one.
        """
        if not isinstance(node, ast.Call):
            return None
        callee:ast.expr = node.func
        if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Name):
            owner:object = self.resolve_global(callee.value.id)
            if owner is None:
                return None
            return library_function_name(getattr(owner, callee.attr, None))
        if isinstance(callee, ast.Name):
            return library_function_name(self.resolve_global(callee.id))
        return None


    def _emit_library_function(self, name:str, args:List[Translated],
                               node:ast.Call)->Translated:
        """A call to a registered device-side library function.

        The device gets the symbol the registry names for this backend; the headers the
        registry asks for are recorded, so the compilation includes them.
        """
        entry:object = library_functions().get(name)
        symbol:Optional[str] = (entry.symbols.get(self.backend) if entry is not None else None)
        if symbol is None:
            available:str = (", ".join(sorted(entry.symbols)) if entry is not None else "")
            self._error(
                f"gpu.{name}() is a registered device function that the {self.backend} "
                f"backend has no implementation for"
                + (f" (it has: {available})" if available else ""), node)
        # The headers the function needs: they go into the generated source, so the symbol
        # the registry names is actually declared.
        for header in entry.headers.get(self.backend, ()):
            if header not in self.device_library_headers:
                self.device_library_headers.append(header)
        arg_codes:List[str] = [code for code, _ in args]
        for code, t in args:
            if is_vec(t):
                self._error(
                    f"gpu.{name}() takes scalars: a registered device function is called "
                    f"as it is declared, and CUDA has no vector overload for it", node)
        # The result type: the first argument's type, unless the function is registered
        # with a different one. A function that takes nothing answers nothing.
        result_t:object = args[0][1] if args else "float"
        if not arg_codes:
            result_t = "void"
        return f"{symbol}({', '.join(arg_codes)})", result_t


    def _emit_gpu_builtin(self, name:str, args:List[Translated], node:ast.Call)->Translated:
        # A complex argument is settled before any table is consulted: those tables are keyed
        # by type, and a complex value has no entry in them (see _emit_complex_builtin).
        if any(self._complex_dtype_of(t) is not None for _, t in args):
            return self._emit_complex_builtin(name, args, node)

        if name in ("min", "max"):
            # pygf's ordering pair, registered: the same emission the bare-name python
            # builtins take, vector expansion included (see _emit_builtin). One name, one
            # code, however the callee was imported.
            return self._emit_builtin(name, args, node)

        if name in INDEX_BUILTINS:
            if len(args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, int):
                self._error(f"gpu.{name}(dim) requires a constant integer dimension (0, 1 or 2)", node)
            if name in ("get_global_id", "get_global_size", "global_id"):
                self.uses_global_id = True
            return emit_index(name, node.args[0].value, self.backend), "int"

        if name == "barrier":
            if args:
                self._error("gpu.barrier() takes no arguments", node)
            if self._metal_syntax:
                from ..metal_codegen import barrier
                return barrier(), "void"
            return emit_barrier(self.backend), "void"

        if name in ATOMIC_BUILTINS:
            # Every atomic is (array, index, value) except atomic_cas, which also takes the
            # value to compare against: (array, index, compare, value).
            expected:int = 4 if name == "atomic_cas" else 3
            if len(args) != expected:
                if name == "atomic_cas":
                    self._error(
                        "gpu.atomic_cas(array, index, compare, value) takes exactly 4 "
                        "arguments", node)
                self._error(f"gpu.{name}(array, index, value) takes exactly 3 arguments", node)
            (arr_code, arr_t), (idx_code, idx_t) = args[0], args[1]
            if isinstance(arr_t, tuple) and arr_t and arr_t[0] == BLOCK_ARRAY:
                # `atomic_add(bins, i, 1)` on this block's own storage: the binding is the
                # declaration, so the array is its C name and its element type.
                arr_code, arr_t = arr_t[2], (PTR, arr_t[1])
            if isinstance(arr_t, tuple) and arr_t and arr_t[0] in WORK_GROUP_VALUES:
                # `atomic_add(bins[blockIdx], i, 1)` on per-group storage: the group's slice
                # is the array the atomic indexes into, and any offset already consumed is
                # folded into the index.
                pending:str = arr_t[3] if len(arr_t) > 3 else ""
                if pending:
                    idx_code = f"({pending}) + ({idx_code})"
                arr_code, arr_t = arr_code, (PTR, arr_t[1])
            val_code, val_t = args[-1]
            compare_code:Optional[str] = (args[2][0] if name == "atomic_cas" else None)
            if not is_ptr(arr_t):
                shared:Optional[Dict[str, object]] = (
                    self.outer_vars.get(node.args[0].id)
                    if isinstance(node.args[0], ast.Name) else None)
                if shared is not None and shared["writable"]:
                    self._error(
                        f"{node.args[0].id} is a single shared value, not an array: update "
                        f"it with `{node.args[0].id} += value` (or -=, &=, |=, ^=), which "
                        f"the translator turns into the device atomic operation, and the "
                        f"result is written back to the enclosing scope after the call. "
                        f"{name}() itself needs an array to index", node)  # type: ignore[union-attr]
                self._error(f"{name}() first argument must be an array", node)
            if isinstance(node.args[0], ast.Name):
                # atomic ops write memory via pointer casts, which static analysis cannot recognize;
                # record the target parameter so PyKernel forcibly marks it writable after compilation
                self.atomic_targets.add(node.args[0].id)
            if arr_t[1] not in self.ATOMIC_ELEMENT_TYPES:
                self._error(
                    f"gpu.{name}() supports int32/uint32/int64/uint64 arrays, got "
                    f"{arr_t[1]}; small integer types (int8/int16) have no device atomic "
                    f"operation on either backend, so use int32 there", node)
            if not is_int(idx_t) or not is_int(val_t):
                self._error(f"gpu.{name}() index and value must be integers", node)
            if name == "atomic_cas" and not is_int(args[2][1]):
                self._error("gpu.atomic_cas() compare value must be an integer", node)
            return emit_atomic(name, arr_code, idx_code, val_code, arr_t[1], self.backend,
                               compare_code,
                               is_local=self._is_local_buffer(node.args[0])), arr_t[1]

        if (name in GENERIC_BUILTINS
                or name in SHARED_MATH_BUILTINS
                or name in SHARED_MATH_TEMPLATES
                or name in CUDA_ONLY_BUILTINS
                or name in CUDA_ONLY_CONVERSIONS):
            return self._emit_generic_builtin(name, args, node)

        self._error(f"unknown gpu builtin: gpu.{name}", node)


    def _emit_cuda_only_builtin(self, name:str, args:List[Translated],
                                node:ast.Call)->Translated:
        """A device intrinsic only the CUDA and HIP backends have.

        The fast approximations, the correctly-rounded forms and the special functions
        CUDA adds have no OpenCL counterpart, so a kernel that uses one is a CUDA or a HIP
        kernel. On OpenCL the call is refused *by name* -- "undeclared identifier" from the
        compiler would not say which of the two the user should reach for.
        """
        if self.launch_backend == "opencl":
            self._error(
                f"gpu.{name}() is a CUDA device intrinsic that the OpenCL backend has no "
                f"counterpart for: the OpenCL kernel language has no such function, so "
                f"this kernel cannot run there. Either specialize the kernel for CUDA or "
                f"HIP, or use one of the functions the shared vocabulary lists "
                f"(see pygpgpu.kernel.builtins)", node)

        # A backend whose tables carry no spelling for this name has no counterpart
        # for it -- Metal's vocabulary is its own, and it has neither the `__expf`
        # family nor the `fast::` names under these spellings. Refusing by name, and
        # naming the backends that do have it, is the whole point of the tables: the
        # device's own "undeclared identifier" would not say where to go instead.
        table:Optional[Dict[str, str]] = (CUDA_ONLY_CONVERSIONS.get(name)
                                          or CUDA_ONLY_BUILTINS.get(name))
        if table is not None and self.launch_backend not in table:
            self._error(
                f"gpu.{name}() is a CUDA device intrinsic that the "
                f"{self.launch_backend} backend has no counterpart for; run this kernel "
                f"on one of {', '.join(sorted(table))}, or compute it from the "
                f"functions all backends have (see pygpgpu.kernel.builtins)", node)

        if name in CUDA_ONLY_CONVERSIONS:
            # A conversion answers an int, so the result type is the int one.
            conversion:str = CUDA_ONLY_CONVERSIONS[name][self.launch_backend]
            expected:int = (1 if name.startswith("float_to_int")
                            else 2 if name.startswith("double_to_int") else 1)
            if len(args) != expected:
                self._error(f"gpu.{name}() takes exactly {expected} argument", node)
            code, t = args[0]
            if is_vec(t):
                self._error(
                    f"gpu.{name}() takes a scalar: CUDA has no vector overload for the "
                    f"conversion, so convert the components one at a time", node)
            # The conversion's own name says what it converts FROM, so the argument is
            # cast to that type first: `__float2int_rn` takes a float, and a double
            # argument would pick the wrong overload.
            source_type:str = ("float" if "float_to_int" in name
                               else "double" if "double_to_int" in name else "int")
            return f"({conversion}(({source_type})({code})))", "int"

        device_name:str = CUDA_ONLY_BUILTINS[name][self.launch_backend]
        # HIP spells the round-toward-zero forms with a device helper rather than an
        # intrinsic (it has no such intrinsic), so the definition goes into the source
        # before the kernel that calls it.
        helper:Optional[str] = self._round_toward_zero_helper(device_name)
        if len(args) not in (1, 2, 3):
            self._error(f"gpu.{name}() takes 1 to 3 arguments", node)
        arg_codes:List[str] = [code for code, _ in args]
        for code, t in args:
            if is_vec(t):
                self._error(
                    f"gpu.{name}() takes a scalar: CUDA spells this intrinsic for scalars "
                    f"only, so apply it per component", node)
        # The argument's type decides the overload: `__expf` takes a float, and a double
        # argument would pick the (different, less accurate) double form.
        argument_type:str = "float"
        for _, t in args:
            if t == "double":
                argument_type = "double"
        if argument_type == "double" and helper is None:
            # The double form of the fast intrinsics is spelled without the `f`.
            device_name = device_name[:-1] if device_name.endswith("f") else device_name
        cast_arguments:List[str] = [f"({argument_type})({code})" for code in arg_codes]
        return f"{device_name}({', '.join(cast_arguments)})", argument_type


    def _round_toward_zero_helper(self, device_name:str)->Optional[str]:
        """The helper a round-toward-zero call needs, or None for an ordinary symbol.

        The helper is emitted once for the whole source, the first time a call needs it.
        """
        if device_name not in HIP_ROUND_TOWARD_ZERO_HELPERS.values():
            return None
        key:str = next(key for key, value in HIP_ROUND_TOWARD_ZERO_HELPERS.items()
                       if value == device_name)
        source:str = HIP_ROUND_TOWARD_ZERO_SOURCES[key]
        if source not in self.device_helpers:
            self.device_helpers.append(source)
        return device_name


    def _emit_shared_math_builtin(self, name:str, args:List[Translated],
                                  node:ast.Call)->Translated:
        """A math function both backends have.

        OpenCL spells it under the plain name and CUDA under the `f`-suffixed one, except
        for the `native_*` family, which CUDA spells with the double-underscore
        intrinsics (that is where the "native" comes from: the device's own instruction).

        A vector argument is expanded per component, the way the ordinary math
        intrinsics are: the device has no vector overload for the scalar-only ones
        (``erf``, ``tgamma``, ...), and the ones that are component-wise (``fract``,
        ``sign``, ``fmin``) work per component as well.
        """
        if len(args) not in (1, 2, 3):
            self._error(f"gpu.{name}() takes 1 to 3 arguments", node)
        arg_codes:List[str] = [code for code, _ in args]
        vec_t:object = next((t for t, _ in args if is_vec(t)), None)
        # The functions that are component-wise: they take a vector and answer one.
        component_wise:bool = name in ("fract", "sign", "fmin", "fmax", "ldexp")
        for code, t in args:
            if is_vec(t) and not component_wise:
                self._error(
                    f"gpu.{name}() takes a scalar: the device has no vector overload for "
                    f"it, so apply it per component", node)
        result_t:object = args[0][1]
        # The functions that answer a different type than they take: the rounding
        # functions answer an int, and the conversions answer one too.
        if name in ("rint", "round", "trunc", "floor", "ceil"):
            result_t = "int"
        if vec_t is not None:
            # The component-wise ones: expand per component and rebuild the vector.
            components:List[str] = _vec_field_names(_vec_registry()[vec_t])
            terms:List[str] = []
            for component in components:
                piece_args:List[str] = [
                    (f"({self._component(code, t, component)})"
                     if is_vec(t) else code)
                    for code, t in args]
                terms.append(self._shared_math_call(name, piece_args, node))
            return self._vec_maker(vec_t, ", ".join(terms)), vec_t
        return self._shared_math_call(name, arg_codes, node), result_t


    def _shared_math_call(self, name:str, arg_codes:List[str],
                          node:ast.Call)->str:
        """One call to a shared math function, in the spelling this backend uses."""
        table:Dict[str, str] = (SHARED_MATH_TEMPLATES[name] if name in SHARED_MATH_TEMPLATES
                                else SHARED_MATH_BUILTINS[name])
        if self.launch_backend not in table:
            # The tables are keyed by backend precisely so a missing spelling cannot
            # be guessed: each one names the callee or the expression that backend
            # actually has. Say which backend and which function, instead of letting
            # a bare KeyError escape from inside the translation.
            self._error(
                f"{name}() has no spelling for the {self.launch_backend} backend; "
                f"run this kernel on one of {', '.join(sorted(table))}, "
                f"or compute it from arithmetic in the body", node)
        if name in SHARED_MATH_TEMPLATES:
            # The backends spell this one differently, so the template says which
            # spelling this backend uses.
            template:str = SHARED_MATH_TEMPLATES[name][self.launch_backend]
            return template.format(code=arg_codes[0])
        device_name:str = SHARED_MATH_BUILTINS[name][self.launch_backend]
        return f"{device_name}({', '.join(arg_codes)})"


    def _emit_complex_builtin(self, name:str, args:List[Translated],
                              node:ast.Call)->Translated:
        """A builtin call with a complex argument: the header's function, or a refusal.

        Both builtin dispatchers reach here first -- `_emit_gpu_builtin` for the package's own
        vocabulary and `_emit_builtin` for python's builtins -- because their tables are keyed
        by *type* and a complex value has no entry in any of them. That is what emitted
        `sqrtf(z)`, `min((int)(z), (int)(w))`, `fminf(fmaxf(z, 0.0f), 1.0f)` and a `round(z)`
        that returned the value unchanged: a call the compiler could not type, or one that
        silently did nothing.

        The split follows numpy, which is the authority the interpreter already runs: a name
        numpy defines for a complex value is answered from the complex header, and a name it
        refuses is refused here by name. Where the two sides disagree, the interpreter wins --
        which is why `round(z)` is refused (python's builtin has no complex form) while
        `np.round(z)` is answered.
        """
        for _code, t in args:
            if is_ptr(t) or is_image(t) or is_vec(t):
                self._error(
                    f"{name}() mixes a complex value with a {t}: a complex value is a scalar, "
                    f"and the device has no complex vector or array element", node)

        dtypes:List[Optional[np.dtype]] = [self._complex_dtype_of(t) for _, t in args]
        sides:List[Optional[str]] = [None if dtype is None else self._complex_side(t, dtype)
                                     for (_, t), dtype in zip(args, dtypes)]
        # A mixed pair (`min(c64, c128)`) is computed at the wider width, which is the rule
        # the arithmetic follows: nothing is ever rounded down.
        target:str = "c128" if "c128" in sides else "c64"
        element:str = "double" if target == "c128" else "float"
        codes:List[str] = []
        for (code, _t), dtype, side in zip(args, dtypes, sides):
            if dtype is None:
                # A real operand: `pow(z, 2)` and `mix(z, w, 0.5)` both have one, and it is
                # built into the complex value it stands for.
                self._require_complex(np.dtype(np.complex128 if target == "c128"
                                               else np.complex64))
                codes.append(f"{target}_from_real(({element})({code}))")
                continue
            self._require_complex(dtype)
            codes.append(code if side == target
                         else self._convert_complex(code, cast(str, side), target))

        if name in COMPLEX_BUILTIN_FUNCTIONS:
            function:str = COMPLEX_BUILTIN_FUNCTIONS[name]
            expected:int = 2 if function == "pow" else 1
            if len(args) != expected:
                self._error(f"{name}() takes exactly {expected} argument(s)", node)
            return f"{target}_{function}({', '.join(codes)})", target

        if name in COMPLEX_BUILTIN_REAL_RESULTS:
            if len(args) != 1:
                self._error(f"{name}() takes exactly 1 argument", node)
            return f"{target}_{COMPLEX_BUILTIN_REAL_RESULTS[name]}({codes[0]})", element

        if name in ("min", "max", "fmin", "fmax"):
            if len(args) != 2:
                self._error(f"{name}() takes exactly 2 arguments", node)
            # `min`/`max` propagate a NaN operand (`np.minimum`'s rule, and python's), while
            # `fmin`/`fmax` ignore it (`np.fmin`'s), so the two families are two functions in
            # the header rather than one with a flag.
            if name in ("fmin", "fmax"):
                pick:str = name
            else:
                pick = "min" if name == "min" else "max"
            return f"{target}_{pick}({codes[0]}, {codes[1]})", target

        if name in ("clamp", "saturate"):
            zero:str = f"{target}_from_real(({element})(0))"
            one:str = f"{target}_from_real(({element})(1))"
            if name == "saturate":
                bounds:List[str] = [zero, one]
            else:
                if len(args) != 3:
                    self._error("clamp(a, lo, hi) takes exactly 3 arguments", node)
                bounds = codes[1:3]
            return f"{target}_clamp({codes[0]}, {bounds[0]}, {bounds[1]})", target

        if name == "mix":
            if len(args) != 3:
                self._error("mix(a, b, t) takes exactly 3 arguments", node)
            # `x * (1 - a) + y * a`, the body pygf's `mix` computes: the blend factor is a
            # complex value here like any other operand.
            one_code:str = f"{target}_from_real(({element})(1))"
            factor:str = f"{target}_sub({one_code}, {codes[2]})"
            return (f"{target}_add({target}_mul({codes[0]}, {factor}),"
                    f" {target}_mul({codes[1]}, {codes[2]}))"), target

        if name == "mad":
            if len(args) != 3:
                self._error("mad(a, b, c) takes exactly 3 arguments", node)
            return f"{target}_add({target}_mul({codes[0]}, {codes[1]}), {codes[2]})", target

        reason:Optional[str] = COMPLEX_BUILTIN_REFUSALS.get(name)
        functions:str = ", ".join(sorted(
            set(COMPLEX_BUILTIN_FUNCTIONS) | set(COMPLEX_BUILTIN_REAL_RESULTS)
            | {"min", "max", "clamp", "mix", "mad"}))
        if reason is not None:
            self._error(f"{name}() has no complex meaning ({reason})", node)
        self._error(
            f"{name}() is not defined for a complex value: the complex operations are "
            f"{functions}", node)
