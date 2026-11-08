from __future__ import annotations
import ast
import inspect
import math
from typing import (Any, Dict, List, Optional, Tuple)
import numpy as np
from .nested_launch import _NestedLaunchMixin
from ..device_builtins import (
    ATOMIC_BUILTINS,
)
from .notation import (
    DeviceType,
    PTR,
    Translated,
)
from .receivers import (
    BUILTIN_METHODS,
    CALL_METHOD,
    CONVERSION_METHODS,
    _receiver_param_of,
    _self_member_call_kind,
    self_attr_kind,
)
from .vectors import (
    _vec_registry,
)
from .predicates import (
    is_ptr,
)
from .walk import (
    _NUMPY_DEVICE_TYPES,
)


class _CallsMixin(_NestedLaunchMixin):

    def _visit_call(self, node:ast.Call)->Translated:
        func:ast.expr = node.func
        if node.keywords:
            # A structure constructor takes member names as keywords -- Ray(origin=p) is
            # how python's ctypes spelling addresses members -- so it is translated before
            # the positional reordering below, which only knows about plain functions.
            if isinstance(func, ast.Name):
                candidate:object = self.resolve_global(func.id)
                if isinstance(candidate, type) and self._is_structure_class(candidate):
                    return self._visit_structure_constructor_call(node, candidate)
                # A plain class the scan synthesized into a structure is a constructor too --
                # it has to be handled *here*, before keyword reordering, because reordering
                # fills a call's omitted arguments from `__init__`'s defaults, and a member
                # whose default is `None` would reach the device as a None constant.
                if isinstance(candidate, type):
                    implied:Optional[np.dtype] = self._structure_from_construction(candidate)
                    if implied is not None:
                        self.plain_class_dtypes.add(implied)
                        return self._visit_structure_constructor_call(node, implied)
            # A helper may be called with keywords; nothing below needs to know, so the
            # call is rewritten to positional order first, using the callee's signature.
            # Builtins take positional arguments only, so they keep refusing.
            node = self._reorder_keywords(node)
            if node.keywords:
                self._error("keyword arguments are not supported for this call", node)

        # A builtin is recognized by the *object* the callee resolves to, never by the
        # spelling of the name, so it does not matter whether it was imported, reached
        # through the package, or is a user function that merely shares the name.
        quaternion_result:Optional[Translated] = self._emit_quaternion_call(node)
        if quaternion_result is not None:
            return quaternion_result

        builtin:Optional[str] = self._builtin_call_name(node)
        if builtin is not None:
            if builtin in ("zeros_per_group", "zeros_per_block", "work_size"):
                self._error(
                    f"{builtin}() is a statement, not a value: use it on its own, as "
                    f"`name = {builtin}(...)`", node)
            # An atomic's index may name several axes (`atomic_add(a, (i, j), v)`), which is
            # the same row-major flattening a subscript uses, so it is flattened **instead of**
            # being visited as an expression: a bare tuple has no device value of its own. A
            # scalar index stays a flat offset, so the 1-D spelling keeps working on a
            # multidimensional array. The array is the first argument, which is where the
            # declared shape comes from.
            args:List[Translated] = []
            for position, argument in enumerate(node.args):
                if (builtin in ATOMIC_BUILTINS and position == 1
                        and isinstance(argument, ast.Tuple)):
                    args.append(self._flatten_index(argument, node.args[0], node))
                else:
                    args.append(self._visit_expression(argument))
            return self._emit_gpu_builtin(builtin, args, node)

        # A registered device-side library function: recognized by the object the callee
        # resolves to, like a builtin, and emitted as the symbol the registry names.
        library:Optional[str] = self._library_call_name(node)
        if library is not None:
            library_args:List[Translated] = [
                self._visit_expression(argument) for argument in node.args]
            return self._emit_library_function(library, library_args, node)

        if isinstance(func, ast.Name):
            name:str = func.id
            if name == "complex":
                # Handled before the arguments are visited: `complex("3+4j")` names a string
                # that has no device value at all, and visiting it first would report
                # "unsupported constant" instead of the reason a caller can act on.
                return self._emit_complex_constructor(node)
            args = [self._visit_expression(a) for a in node.args]
            arg_codes:List[str] = [c for c, _ in args]
            # `v(args)` on a structure value is `v.__call__(args)`: the callee is a value,
            # not a function, and a class that defines `__call__` gives it a meaning.
            holder:DeviceType = self._lookup(name)
            if holder is not None and self._is_structure_operand(holder):
                owner:Optional[Tuple[type, Any]] = self._magic_owner(holder, CALL_METHOD)
                if owner is not None:
                    _cls, method = owner
                    return self._emit_device_call(
                        CALL_METHOD, method, [(name, holder)] + list(args), node,
                        self_param=_receiver_param_of(method), receiver_as_argument=True)

            if name in ("int", "float", "bool"):
                if len(args) != 1:
                    self._error(f"{name}() takes exactly 1 argument", node)
                # `int(box)` is `box.__int__()`, the way `abs(box)` is `box.__abs__()`:
                # a class that defines the conversion gives it a meaning, and a device
                # keeps it rather than casting the aggregate.
                converted:Optional[Translated] = self._emit_magic_conversion(
                    CONVERSION_METHODS[name], arg_codes[0], args[0][1], node)
                if converted is not None:
                    return converted
                t:DeviceType = {"int": "int", "float": "float", "bool": "bool"}[name]
                return f"(({t})({arg_codes[0]}))", t

            if name in ("abs", "min", "max", "round", "pow"):
                # `abs(box)` and `round(box)` are single-argument calls on the class;
                # `min`/`max`/`pow` take two and are not method-shaped.
                if name in ("abs", "round") and len(args) == 1:
                    converted = self._emit_magic_conversion(
                        BUILTIN_METHODS[name], arg_codes[0], args[0][1], node)
                    if converted is not None:
                        return converted
                return self._emit_builtin(name, args, node)

            if name == "len":
                return self._emit_len(args, node)

            vec_name:Optional[str] = self._vec_type_name(name)
            if vec_name is not None:
                return self._emit_vec_constructor(vec_name, args, node)
            reason:Optional[str] = self._unsupported_vec_reason(self.resolve_global(name))
            if reason is not None:
                self._error(reason, node)

            target:object = self.resolve_global(name)
            if isinstance(target, type):
                # A `ctypes.Structure` subclass is not an object in the python sense: it is a
                # layout, and a device type is generated from it, so `Ray(origin, direction)`
                # is a structure value -- a compound literal keyed by member name. Every
                # other class stays refused: there is nothing on the device to instantiate.
                from .. import structs

                structure:Optional[np.dtype] = structs.structure_dtype_of(target)
                if structure is not None:
                    return self._emit_structure_constructor(structs.register(structure),
                                                            structure, args, node)
                # A plain class whose value the body builds and then reads is a layout too,
                # decided by what the body does with it: the members it reads and the methods
                # it calls, typed from the arguments at this call site. Nothing here parses
                # the class's `__init__` or its annotations.
                implied:Optional[np.dtype] = self._structure_from_construction(target)
                if implied is not None:
                    self.plain_class_dtypes.add(implied)
                    return self._emit_structure_constructor(structs.register(implied),
                                                            implied, args, node)
                self._error(
                    f"'{name}' is a class: device code has no objects, so it cannot be "
                    f"constructed inside a kernel. Keep the values in separate scalar or "
                    f"vector variables, or pass a ctypes.Structure as a kernel argument",
                    node)
            if callable(target):
                return self._emit_device_call(name, target, args, node)
            self._error(f"unknown function '{name}'", node)

        if isinstance(func, ast.Attribute):
            receiver_name:Optional[str] = (func.value.id
                                           if isinstance(func.value, ast.Name) else None)
            if (receiver_name is not None and self.self_param
                    and receiver_name == self.self_param):
                args:List[Translated] = [self._visit_expression(a) for a in node.args]
                # The method belongs to the class the *receiver* is, not necessarily to the
                # kernel's own class: inside `Camera.render`, `scene.ray_color(...)` reaches
                # a Scene method, and a helper bound to a Scene receiver calls Scene's own
                # methods through `self.`. The receiver's dtype names that class.
                from .. import structs as _structs_module  # noqa: PLC0415

                receiver_t:DeviceType = self._lookup(self.self_param)
                receiver_owner:Optional[type] = None
                if isinstance(receiver_t, tuple) and is_ptr(receiver_t):
                    dtype:Optional[np.dtype] = _structs_module.dtype_for(receiver_t[1])
                    if dtype is not None:
                        receiver_owner = _structs_module.declared_class(dtype)
                call_owner:Optional[type] = receiver_owner or self.self_owner
                kind:str
                target:object
                kind, target = _self_member_call_kind(call_owner, func.attr)
                if kind in ("method", "staticmethod") and receiver_owner is not None \
                        and call_owner is not self.self_owner:
                    # A method of the receiver's class: the receiver is this call's first
                    # argument, so it binds positionally, exactly as `ray.at(t)` does.
                    return self._emit_device_call(
                        func.attr, target,
                        [(self.self_param, receiver_t)] + args, node,
                        self_param=(target.__code__.co_varnames[0]
                                    if getattr(target, "__code__", None) else "self"),
                        receiver_as_argument=True)
                if kind in ("method", "staticmethod"):
                    return self._emit_self_method_call(kind, target, func.attr, args, node)
                self._error(
                    f"calling self.{func.attr}() from device code is not supported: "
                    f"self.{func.attr} is {self._why_not_callable(kind, func.attr)}", node)
            module:object = (self.resolve_global(receiver_name)
                          if receiver_name is not None else None)
            if module is math or module is np:
                if module is np and func.attr in ("clip", "where"):
                    # np.clip/np.where read their own arguments: a None bound means "no
                    # bound" and must not be evaluated as a device expression.
                    return self._emit_numpy_elementwise_nodes(func.attr, node)
                args = [self._visit_expression(a) for a in node.args]
                if module is np:
                    if func.attr in _NUMPY_DEVICE_TYPES:
                        # np.float32(x): a cast, exactly what `(float)x` means on the device.
                        return self._emit_numpy_cast(func.attr, args, node)
                    elementwise:Optional[Translated] = self._emit_numpy_elementwise(
                        func.attr, args, node)
                    if elementwise is not None:
                        return elementwise
                return self._emit_math_call(func.attr, args, node, module_name=receiver_name or "")
            if isinstance(module, type):
                # Class.static_method(...): a class name is not a device value, so the
                # only reachable form is a staticmethod, resolved to its plain function.
                kind, target = self_attr_kind(module, func.attr)
                if kind in ("staticmethod", "method"):
                    # A plain function taken off its class is just a function; the
                    # receiver of "method" stays an explicit first argument.
                    args = [self._visit_expression(a) for a in node.args]
                    return self._emit_device_call(func.attr, target, args, node)
                self._error(
                    f"cannot call {receiver_name}.{func.attr}() from device code: "
                    f"only static methods of a class are usable there, and "
                    f"{receiver_name}.{func.attr} is {self._why_not_callable(kind, func.attr)}", node)
            if inspect.ismodule(module) and callable(getattr(module, func.attr, None)):
                # module.function(...): a plain helper reached through its module.
                args = [self._visit_expression(a) for a in node.args]
                return self._emit_device_call(func.attr, getattr(module, func.attr), args, node)
            # Anything else is a receiver that is a *value*: `ray.hit_sphere(...)` on a
            # structure, or a method reached through one of its members.
            return self._emit_method_on_value(func, node)

        # `inner[(n,)](args)`: one work item launches another kernel. Left until here so a
        # structure constructor and a helper's keyword call are already resolved.
        if self._emit_nested_launch(node):
            return "", "void"

        self._error("unsupported call expression", node)


    def _emit_method_on_value(self, func:ast.Attribute, node:ast.Call)->Translated:
        """`value.method(args)` as a device call whose first argument is the value.

        A structure's methods are ordinary python functions defined in the class the caller
        declared, and their first parameter is the receiver -- which is exactly what `self`
        already is in a device helper. So there is nothing new to translate: the call becomes
        a device function whose first argument is the receiver expression, and the body then
        reads `self.origin` the way any structure-typed parameter would.

        The class has to be reachable, which is why only a `ctypes.Structure` subclass
        qualifies: a bare record dtype has no methods, and two classes with one layout share
        a dtype, so the dtype alone cannot say whose methods to compile.
        """
        from .. import structs

        base_code, base_t = self._visit_expression(func.value)
        # A complex value's methods are numpy's, and the device header spells the same
        # operations: `.conjugate()` flips the imaginary part, and the rest are the free
        # functions a reader may reach for through the value itself.
        complex_dtype:Optional[np.dtype] = self._complex_dtype_of(base_t)
        if complex_dtype is not None:
            translated:Optional[Translated] = self._emit_complex_method(
                func.attr, base_code, base_t, complex_dtype, node)
            if translated is not None:
                return translated
            self._error(
                f"a complex value has no method '.{func.attr}()': numpy's complex scalars"
                f" define 'conjugate()' and 'conj()', and the rest are the module functions"
                f" (np.sqrt, np.abs, np.angle, ...)", node)
        receiver_is_pointer:bool = (isinstance(base_t, tuple) and is_ptr(base_t))
        base_struct:str = base_t[1] if receiver_is_pointer else base_t
        owner:Optional[type] = None
        if isinstance(base_struct, str):
            dtype:Optional[np.dtype] = structs.dtype_for(base_struct)
            if dtype is not None:
                # The class that actually defines the method: two callers can declare
                # classes with one layout, and only one of them carries this attribute.
                owner = (structs.declared_class_carrying(dtype, func.attr)
                         or structs.declared_class(dtype))
            elif base_struct in _vec_registry():
                owner = _vec_registry()[base_struct]
        if owner is None:
            self._error(
                f"cannot call '.{func.attr}()' on a value of type {base_t}: methods are "
                f"translated from the class they are defined on, and only a "
                f"ctypes.Structure subclass or a vector type carries one", node)

        kind:str
        target:object
        kind, target = self_attr_kind(owner, func.attr)
        if kind == "method":
            args = [self._visit_expression(a) for a in node.args]
            # A method that assigns to its receiver's members needs the receiver *by
            # address*: a by-value structure is a copy, so `self.normal = ...` inside the
            # method would be lost. `HitRecord.set_face_normal` is exactly that shape, and
            # with a by-value receiver every normal stayed zero -- the GPU drew a uniform
            # grey ball while the interpreter, which runs the method unchanged, was right.
            receiver_param:str = (target.__code__.co_varnames[0]
                                  if getattr(target, "__code__", None) else "self")
            if receiver_is_pointer:
                # A structure argument already travels as a pointer: hand it over as it is,
                # whether the callee writes its members or only reads them. The receiver's
                # parameter name is named so the callee binds it as the structure it is --
                # without it, `self.origin` inside the method would read the value form.
                return self._emit_device_call(
                    func.attr, target, [(base_code, base_t)] + args, node,
                    self_param=receiver_param, receiver_as_argument=True)
            # A value receiver is handed over by address: every structure parameter is a
            # pointer (so `self.<member>` reads the one form), and a local value's address
            # is available at the call site. The copy python makes for a by-value call is
            # the one thing that cannot be kept, and a method that assigns to its receiver
            # needs the address for the write to reach the caller.
            return self._emit_device_call(
                func.attr, target,
                [(f"&({base_code})", (PTR, base_struct))] + args, node,
                self_param=receiver_param, receiver_as_argument=True)
        if kind == "staticmethod":
            args = [self._visit_expression(a) for a in node.args]
            return self._emit_device_call(func.attr, target, args, node)
        self._error(
            f"cannot call '.{func.attr}()' on {owner.__name__}: it is "
            f"{self._why_not_callable(kind, func.attr)}", node)


    def _reorder_keywords(self, node:ast.Call)->ast.Call:
        """Rewrite a call with keyword arguments into positional order.

        Only a plain-python helper has a signature to bind against; anything else
        (builtin, gpu.*, vector constructor) keeps its keywords so the caller can
        report them. Defaults are filled in, matching what the host does for a kernel
        call, so the device function always receives one value per parameter.
        """
        func:ast.expr = node.func
        target:object = None
        if isinstance(func, ast.Name):
            target = self.resolve_global(func.id)
            if not callable(target) or getattr(target, "is_pykernel", False):
                return node
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
              and not (self.self_param and func.value.id == self.self_param)):
            module:object = self.resolve_global(func.value.id)
            target = getattr(module, func.attr, None)
            if not callable(target):
                return node
        else:
            return node

        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return node

        positional:List[ast.expr] = list(node.args)
        supplied:Dict[str, ast.expr] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                return node
            supplied[keyword.arg] = keyword.value

        ordered:List[ast.expr] = []
        for index, (name, parameter) in enumerate(signature.parameters.items()):
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            if index < len(positional):
                if name in supplied:
                    self._error(
                        "the argument '%s' is given both positionally and by keyword"
                        % name, node)
                ordered.append(positional[index])
            elif name in supplied:
                ordered.append(supplied[name])
            elif parameter.default is not parameter.empty:
                ordered.append(ast.Constant(value=parameter.default))
            else:
                self._error("missing required argument '%s'" % name, node)

        unknown = set(supplied) - set(signature.parameters)
        if unknown:
            self._error(
                "unknown keyword argument%s: %s"
                % ("" if len(unknown) == 1 else "s", ", ".join(sorted(unknown))), node)

        return ast.Call(func=node.func, args=ordered, keywords=[])
