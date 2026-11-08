from __future__ import annotations
import ast
import enum
import math
from typing import (Any, Dict, Optional, Tuple)
import numpy as np
from ..device_builtins import (
    INDEX_VECTOR_AXES_CUDA,
    INDEX_VECTOR_AXES_OPENCL,
    INDEX_VECTOR_BUILTINS,
)
from .notation import (
    DeviceType,
    Translated,
)
from .receivers import (
    _enum_member_value,
    _self_member_call_kind,
)
from .vectors import (
    _C_FIELD_TYPES,
    _vec_component_name,
    _vec_registry,
    is_vec,
)
from .builtins import (
    _MATH_CONSTS,
)
from .predicates import (
    is_ptr,
)
from .walk import (
    _NUMPY_DEVICE_TYPES,
)


class _AttributesMixin:

    def _visit_attribute(self, node:ast.Attribute)->Translated:
        # self.xxx -> the receiver struct's member, or an extra parameter _self_xxx for an
        # ndarray attribute (the read/write analysis and the shape plumbing need it apart).
        if (self.self_param and isinstance(node.value, ast.Name) and node.value.id == self.self_param):
            if node.attr not in self.self_struct_members and node.attr not in \
                    self.self_attr_types:
                # The receiver may be a structure pointer instead of a field set
                # (`def hit(self, ...)` called with a Scene): its scalar members are
                # fields of that pointer.
                from .. import structs as _structs_module  # noqa: PLC0415

                pointer_t:DeviceType = self._lookup(self.self_param)
                if isinstance(pointer_t, tuple) and is_ptr(pointer_t):
                    pointer_dtype:Optional[np.dtype] = _structs_module.dtype_for(
                        pointer_t[1])
                    if pointer_dtype is not None:
                        pointer_members:Dict[str, object] = _structs_module.member_types(
                            pointer_dtype, self.backend)
                        if node.attr in pointer_members:
                            return (f"({self.self_param})->{node.attr}",
                                    pointer_members[node.attr])
            if node.attr in self.self_struct_members:
                from .. import structs as _structs_module  # noqa: PLC0415

                member:str = self.self_struct_members[node.attr]
                struct_dtype:Optional[np.dtype] = _structs_module.dtype_for(
                    self.self_struct_type[1])
                members:Dict[str, object] = _structs_module.member_types(
                    struct_dtype, self.backend)
                if node.attr not in members:
                    self._error(
                        f"structure {self.self_struct_type[1]} has no member '.{node.attr}' "
                        f"(available: {', '.join(members)})", node)
                # The receiver struct travels as a pointer, so the member read goes
                # through '->'.
                return (f"({self.self_param})->{member}", members[member])
            if node.attr not in self.self_attr_types:
                # A receiver that arrived as this function's own first argument
                # (`def at(self, t)` called as `ray.at(t)`) or as one structure pointer
                # (`def hit(self, ...)` called with a Scene) is a structure pointer with no
                # field set: read its members through the pointer, like any other
                # structure parameter.
                receiver_t:DeviceType = self._lookup(self.self_param)
                if isinstance(receiver_t, tuple) and is_ptr(receiver_t):
                    from .. import structs as _structs_module  # noqa: PLC0415

                    dtype:Optional[np.dtype] = _structs_module.dtype_for(receiver_t[1])
                    if dtype is not None:
                        members:Dict[str, object] = _structs_module.member_types(
                            dtype, self.backend)
                        if node.attr in members:
                            return (f"({self.self_param})->{node.attr}" ,
                                    members[node.attr])
                kind:str
                _target:object
                kind, _target = _self_member_call_kind(self.self_owner, node.attr)
                if kind in ("method", "staticmethod"):
                    # A helper of the kernel's own class: call it instead of reading it,
                    # which is the one supported form (a function value has no device type).
                    self._error(
                        f"self.{node.attr} is a function; call it as self.{node.attr}(...) "
                        f"instead of reading it", node)
                self._error(
                    f"self.{node.attr} was not discovered on this kernel "
                    f"(discovered attributes: {', '.join(self.self_attr_types) or 'none'})", node)
            return f"_self_{node.attr}", self.self_attr_types[node.attr]

        if isinstance(node.value, ast.Name):
            module:object = self.resolve_global(node.value.id)
            # `Kind.B`: reading a member off an enum *class* is that member's value, exactly
            # as reading the bare name `B` would be. The class itself has no device
            # representation, so this has to be answered before it reaches the generic
            # attribute path.
            #
            # The base has to be a real enum *class* before the attribute is read: `blockIdx`
            # is an `_IndexVector` whose attributes compute the launch's numbers, so an
            # unconditional `getattr` here asked for a work item that does not exist yet and
            # made every `blockIdx.x` fail with "get_group_id() is only available inside a
            # @kernel function body".
            if isinstance(module, enum.EnumMeta):
                enum_member:Optional[Tuple[Any, Any]] = _enum_member_value(
                    getattr(module, node.attr, None))
                if enum_member is not None:
                    return self._visit_enum_constant(enum_member, node)
            if module is math or module is np:
                if node.attr in _MATH_CONSTS:
                    return _MATH_CONSTS[node.attr], "float"
                if module is np and node.attr in _NUMPY_DEVICE_TYPES:
                    # np.float32 / np.int32 / ... name a type, not a value. Reading it as a
                    # value has no device meaning; a cast is `x.astype(...)` or `np.float32(x)`,
                    # and a dtype keyword is consumed by the creation this feature lifts out.
                    self._error(
                        f"np.{node.attr} is a type: use it as a keyword (dtype=np.{node.attr}) "
                        f"or as a constructor (np.{node.attr}(value)), not as a value", node)
                self._error(f"{node.value.id}.{node.attr} is not supported", node)

        base_code, base_t = self._visit_expression(node.value)
        # A complex value's members are its real and imaginary parts, which the header
        # spells as its own accessors: the structure's members are `re`/`im` on the device
        # and `.real`/`.imag` in numpy, and a reader should never have to know which.
        complex_dtype:Optional[np.dtype] = self._complex_dtype_of(base_t)
        if complex_dtype is not None:
            kind:str = self._complex_side(base_t, complex_dtype)
            member:str = {"real": "real", "imag": "imag", "re": "real", "im": "imag"}.get(
                node.attr, "")
            if member:
                self._require_complex(complex_dtype)
                element:str = "double" if kind == "c128" else "float"
                return f"{kind}_{member}({base_code})", element
            self._error(
                f"a complex value has no member '.{node.attr}': use '.real', '.imag', or"
                f" call '.conjugate()'", node)
        if is_vec(base_t):
            component:str = _vec_component_name(node.attr)
            if len(component) > 1:
                return self._emit_vec_swizzle(base_code, base_t, component, node)
            field_key:(str, str) = (base_t, component)
            if field_key not in _C_FIELD_TYPES:
                self._error(
                    f"vector type {base_t} has no component '.{node.attr}'", node)
            # `threadIdx.x` is one launch term, not a component of a built vector: the *name*
            # resolves to a whole `int3` (`make_int3(...)` on CUDA, a compound literal on
            # OpenCL), so reading one component off it rebuilt all three and then indexed
            # them. The scalar spelling is what the axis means, and it is what the OpenCL
            # path already answers for `get_local_id(0)`.
            axis:Optional[str] = self._index_vector_axis(node, base_t)
            if axis is not None:
                return axis, _C_FIELD_TYPES[field_key]
            return f"({base_code}.{component})", _C_FIELD_TYPES[field_key]

        # A structure member, read with python's attribute spelling. It cannot be confused
        # with a vector component: the branch above has already claimed every base that is a
        # vector, so a name reaching here is only a member if the base is a structure.
        from .. import structs

        if isinstance(base_t, tuple) and is_ptr(base_t):
            # `p.x` on a structure pointer: a structure argument always travels as one, so
            # the member read goes through '->'. A record array (a written object argument)
            # reads its element first, and a receiver passed by address (`&record`) is
            # addressed as `(&record)->x`.
            dtype:Optional[np.dtype] = structs.dtype_for(base_t[1])
            if dtype is not None:
                # A `@property` is a member the class computes: `box.total` is a call to
                # its getter, not a field read. It is tried before the field lookup, and
                # only for a name that is not a real member -- a property name that
                # shadows a field would be a field read in python too, because the class
                # body's descriptor replaces the field of the same name.
                property_read:Optional[Translated] = self._emit_property_read(
                    structs.declared_classes(dtype), node.attr, (base_code, base_t), node)
                if property_read is not None:
                    return property_read
                members:Dict[str, object] = structs.member_types(dtype, self.backend)
                if node.attr not in members:
                    self._error(
                        f"structure {base_t[1]} has no member '.{node.attr}' "
                        f"(available: {', '.join(members)})", node)
                if base_code.startswith("&"):
                    # A receiver passed by address: no record array around it.
                    return f"({base_code})->{node.attr}", members[node.attr]
                if structs.is_declared_structure(np.dtype(dtype)) and \
                        not base_code.endswith(']'):
                    # A structure pointer parameter: addressed directly.
                    return f"({base_code})->{node.attr}", members[node.attr]
                return f"({base_code}[0]).{node.attr}", members[node.attr]

        if isinstance(base_t, str):
            dtype:Optional[np.dtype] = structs.dtype_for(base_t)
            if dtype is not None:
                members:Dict[str, object] = structs.member_types(dtype, self.backend)
                if node.attr not in members:
                    self._error(
                        f"structure {base_t} has no member '.{node.attr}' "
                        f"(available: {', '.join(members)})", node)
                return f"({base_code}).{node.attr}", members[node.attr]

        self._error(f"unsupported attribute access: .{node.attr}", node)


    def _index_vector_axis(self, node:ast.Attribute, vector_type:str)->Optional[str]:
        """The scalar expression for `blockIdx.x`-style axis, or None when it is not one.

        The index vectors are *values* in the source (`buf[blockIdx][j]`), so their name
        resolves to a built vector. Taking a component off that rebuilt all three terms and
        indexed one: `int i = make_int3((int)threadIdx.x, ...).x`. The axis itself is a single
        launch term, and this answers it directly -- the same number the `get_*` spelling
        gives, which is what the axis meant in the first place.
        """
        if len(node.attr) != 1:
            return None
        base:ast.expr = node.value
        if not isinstance(base, ast.Name) or base.id not in INDEX_VECTOR_BUILTINS:
            return None
        if INDEX_VECTOR_BUILTINS[base.id][0] != vector_type:
            return None
        component:str = _vec_component_name(node.attr)
        axis:int = "xyz".find(component)
        if axis < 0:
            return None
        table:Dict[str, Tuple[str, str, str]] = (
            INDEX_VECTOR_AXES_OPENCL if self._opencl_syntax else INDEX_VECTOR_AXES_CUDA)
        return table[base.id][axis]


    def _unsupported_vec_reason(self, target:object)->Optional[str]:
        """Why a `pygf` class has no device vector type, or None when it has one.

        The alias classes with a device counterpart resolve through `_vec_type_name`. The
        rest are fp16 vectors (which the backends do not take) and matrices (which device
        code has no type for), and they deserve a message that says so instead of the
        generic "not a function that can be compiled".
        """
        if not isinstance(target, type):
            return None
        if not str(getattr(target, "__module__", "")).startswith("pygf"):
            return None
        name:str = target.__name__
        if name.startswith(("matrix", "frame")):
            return (f"{name} is a matrix type: device code has no matrix type, so build the "
                    f"components explicitly")
        for base in target.__mro__:
            if base.__name__.startswith("half"):
                return (f"{name} holds 16-bit floats: the device backends do not support "
                        f"fp16 vectors, use a float or double vector")
        return None


    def _vec_type_name(self, name:str)->Optional[str]:
        """The device vector type a constructor name stands for, or None.

        A vector type is spelled the same way on both sides, but the host side has more
        names for one type: `pygf` gives colour/normal/point/vector aliases that all subclass
        the same vector (`color3f` -> `float3`). Walking the class's bases finds the device
        name, so `color3f(r, g, b)` compiles into the `float3` constructor the body means.
        """
        if name in _vec_registry():
            return name
        target:object = self.resolve_global(name)
        if not isinstance(target, type):
            return None
        for base in target.__mro__:
            if base.__name__ in _vec_registry():
                return base.__name__
        return None
