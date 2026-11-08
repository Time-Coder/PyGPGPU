from __future__ import annotations
import ast
from typing import (Dict, List, Optional)
import numpy as np
from .notation import (
    Translated,
)
from .vectors import (
    _vec_field_names,
    is_vec,
)
from .predicates import (
    is_ptr,
)


class _StructuresMixin:

    def _structure_from_construction(self, target:type)->Optional[np.dtype]:
        """The structure a value built by this class call should become, or None.

        The members come from the body -- what it reads off the value and which methods it
        hands it to -- and the types from the constructor's own arguments. That work happens
        once, before the translator runs, in the same scan that handles an object argument;
        this only looks the answer up.
        """
        from .. import objects  # noqa: PLC0415

        dtype:Optional[np.dtype] = objects.constructed_structure(target)
        if dtype is None:
            return None
        return dtype


    def _is_structure_class(self, candidate:object)->bool:
        """Whether a global name is a `ctypes.Structure` subclass with a device layout."""
        from .. import structs  # noqa: PLC0415

        return structs.structure_dtype_of(candidate) is not None


    def _visit_structure_constructor_call(self, node:ast.Call,
                                          candidate:object)->Translated:
        """A structure constructor called with keywords: `Ray(origin=p, direction=d)`.

        A member name is a keyword, exactly as python's ctypes spelling addresses one, and
        the positional arguments come first in declaration order -- the same shape
        `ctypes.Structure.__init__` accepts.

        ``candidate`` is either a declared class or the dtype a plain class was synthesized
        into. For the latter the positional arguments are matched through the class's
        constructor parameters, not the field order: the fields are sorted by name, so the
        first argument is not necessarily the first field.
        """
        from .. import objects, structs  # noqa: PLC0415

        parameters:List[str] = []
        if isinstance(candidate, np.dtype):
            # A plain class the scan synthesized: the dtype IS the layout -- there is no
            # ctypes.Structure behind it to derive one from.
            dtype:Optional[np.dtype] = candidate
            owner:Optional[type] = structs.declared_class(dtype)
            if owner is not None:
                parameters = objects.constructed_parameters(owner)
        else:
            dtype = structs.structure_dtype_of(candidate)
            parameters = list(dtype.names or ()) if dtype is not None else []
        if dtype is None:
            self._error(f"'{ast.unparse(node.func)}' is not a structure type", node)
        order:List[str] = list(dtype.names or ())
        args:List[Translated] = [self._visit_expression(a) for a in node.args]
        supplied:Dict[str, Translated] = {}
        for position, value in enumerate(args):
            if position >= len(parameters):
                self._error(
                    f"{ast.unparse(node.func)}() takes at most {len(parameters)} positional "
                    f"arguments ({', '.join(parameters)}), got {position + 1}", node)
            supplied[parameters[position]] = value
        for keyword in node.keywords:
            if keyword.arg is None:
                self._error("`**kwargs` cannot construct a structure", node)
            if keyword.arg not in order:
                self._error(
                    f"{ast.unparse(node.func)} has no member '{keyword.arg}' "
                    f"(available: {', '.join(order)})", keyword.value)
            if keyword.arg in supplied:
                self._error(
                    f"member '{keyword.arg}' supplied twice in {ast.unparse(node.func)}()",
                    keyword.value)
            supplied[keyword.arg] = self._visit_expression(keyword.value)
        # Hand the constructor the values keyed by member, so a member given as a keyword
        # does not shift the positional ones; omitted members are zeroed there.
        return self._emit_structure_constructor(structs.register(dtype), dtype,
                                                args, node, supplied=supplied)


    def _emit_structure_constructor(self, name:str, dtype:np.dtype,
                                    args:List[Translated], node:ast.expr,
                                    supplied:Optional[Dict[str, Translated]] = None
                                    )->Translated:
        """`Ray(origin, direction)` as a compound literal keyed by member name.

        A `ctypes.Structure` class describes a layout, so a call to it is a structure *value*
        -- what `np.zeros(1, dtype=Ray)` builds, without the array around it. The arguments
        are positional in declaration order, which is also the order python's
        `ctypes.Structure.__init__` assigns them.
        """
        from .. import structs

        if dtype not in self.private_structs:
            # The typedef has to be emitted even when the structure never appears in the
            # signature: a body that builds one locally still needs the type declared.
            self.private_structs.append(dtype)
        members:Dict[str, object] = structs.member_types(dtype, self.backend)
        order:List[str] = list(dtype.names or ())
        if supplied is None:
            supplied = {}
            # For a *declared* structure the layout is fixed, so too many arguments is the
            # clear error it always was. For a structure a body's plain class was synthesized
            # into, the members are what the body reads -- `Ray(origin, direction)` builds
            # two, and a body reading only `origin` describes a one-member structure -- so
            # the unread arguments are dropped, the way python itself leaves an attribute
            # nobody mentions unassigned.
            parameters:List[str] = order
            if dtype in self.plain_class_dtypes:
                owner:Optional[type] = structs.declared_class(dtype)
                if owner is not None:
                    from .. import objects  # noqa: PLC0415

                    parameters = objects.constructed_parameters(owner) or order
                if len(args) > len(parameters):
                    args = args[:len(parameters)]
            elif len(args) > len(order):
                self._error(
                    f"{name}() takes at most {len(order)} arguments "
                    f"({', '.join(order)}), got {len(args)}", node)
            for position, (code, t) in enumerate(args):
                member:str = parameters[position]
                supplied[member] = (code, t)

        parts:List[str] = []
        positional_parts:List[str] = []
        for member in order:
            if member not in supplied:
                # A missing member is zero, which is what a C aggregate does with an
                # unmentioned field; it keeps `Ray(origin)` working. CUDA/HIP are C++ and
                # cannot skip a field in a positional initialiser, so they spell the zero
                # out; OpenCL uses designated initialisers and simply omits it.
                positional_parts.append(self._zero_value(members[member]))
                continue
            code, t = supplied[member]
            converted:str = self._structure_member_value(name, member, members[member],
                                                         code, t, node)
            parts.append(f".{member} = {converted}")
            positional_parts.append(converted)

        if self._opencl_syntax:
            # OpenCL C is C99, so a designated initialiser names the members it sets.
            return f"(({name}){{{', '.join(parts)}}})", name
        # CUDA and HIP compile as C++, where designated initialisers arrived only in C++20
        # (NVRTC is older); a braced aggregate takes its members in declaration order.
        return f"({name}{{{', '.join(positional_parts)}}})", name


    def _zero_value(self, member_type:object)->str:
        """The zero of a member's device type, for an omitted constructor argument."""
        from .. import structs

        if is_ptr(member_type):
            return "0"
        if is_vec(member_type):
            return self._vec_maker(member_type,
                                   ", ".join("0" for _ in _vec_field_names(member_type)))
        if isinstance(member_type, str) and structs.dtype_for(member_type) is not None:
            dtype:object = structs.dtype_for(member_type)
            inner:str = ", ".join(self._zero_value(structs.member_types(dtype, self.backend)[m])
                                  for m in dtype.names or ())
            return f"({member_type}{{{inner}}})"
        return f"(({self._device_type(member_type)})(0))"


    def _structure_member_value(self, struct_name:str, member:str, member_type:object,
                                code:str, t:object, node:ast.expr)->str:
        """Coerce one constructor argument to the member's own device type."""
        from .. import structs

        if is_ptr(member_type):
            self._error(
                f"{struct_name}(...): member '{member}' is an array; assign its elements "
                f"after construction instead", node)
        if is_vec(member_type):
            if t != member_type:
                self._error(
                    f"{struct_name}(...): member '{member}' is a {member_type}, got {t}", node)
            return code
        if isinstance(member_type, str) and structs.dtype_for(member_type) is not None:
            if t != member_type:
                self._error(
                    f"{struct_name}(...): member '{member}' needs a structure value, "
                    f"got {t}", node)
            return code
        if t == member_type:
            return code
        if member_type == "uchar" and self._member_is_bool(struct_name, member):
            # A bool member is stored as a byte, so assigning a float to it truncates --
            # `front_face = 0.99` became 0, and the branch reading it took the wrong way,
            # which showed as a missing colour channel. The conversion has to be boolean.
            return f"(({code}) != 0)"
        return f"(({self._device_type(member_type)})({code}))"


    def _member_is_bool(self, struct_name:str, member:str)->bool:
        """Whether a structure member's layout type is a bool."""
        from .. import structs  # noqa: PLC0415

        dtype:object = structs.dtype_for(struct_name)
        if dtype is None or member not in (dtype.names or ()):
            return False
        return np.dtype(dtype.fields[member][0]).name == "bool"


    def structure_dtypes(self)->List[object]:
        """Every structure dtype this translation mentions, in no particular order.

        The caller emits one typedef block for the whole source, so the kernel and each
        helper contribute their own set and duplicates are dropped.
        """
        from .. import structs

        dtypes:List[object] = []
        for name, t in list(self.params):
            candidate:object = t[1] if is_ptr(t) else t
            dtype:object = structs.dtype_for(candidate) if isinstance(candidate, str) else None
            if dtype is not None and dtype not in dtypes:
                dtypes.append(dtype)
        for dtype in self.private_structs:
            if dtype not in dtypes:
                dtypes.append(dtype)
        for entry in (self.return_layout or []):
            candidate = entry["type"]
            dtype = structs.dtype_for(candidate) if isinstance(candidate, str) else None
            if dtype is not None and dtype not in dtypes:
                dtypes.append(dtype)
        return dtypes


    def render_structure_typedefs(self)->str:
        """The `typedef`s for every structure this kernel's signature mentions.

        A structure is described by its numpy dtype, so the device type is generated rather
        than declared by the user. Nested structures come first (`render_structures` orders
        them), and a member that is a vector already has its device spelling.
        """
        from .. import structs

        dtypes:List[object] = []
        for name, t in list(self.params):
            for candidate in (t[1] if is_ptr(t) else t,):
                dtype:object = structs.dtype_for(candidate) if isinstance(candidate, str) else None
                if dtype is not None and dtype not in dtypes:
                    dtypes.append(dtype)
        for dtype in self.private_structs:
            if dtype not in dtypes:
                dtypes.append(dtype)
        for entry in (self.return_layout or []):
            candidate:object = entry["type"]
            dtype = structs.dtype_for(candidate) if isinstance(candidate, str) else None
            if dtype is not None and dtype not in dtypes:
                dtypes.append(dtype)
        return structs.render_structures(dtypes, self.backend)
