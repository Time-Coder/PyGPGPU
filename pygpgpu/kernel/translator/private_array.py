# A creation below the magic loop: a private C array per work item.
from __future__ import annotations
import ast
from typing import (Any, List, Optional, Tuple)
import numpy as np
from .type_map import (
    DTYPE_MAP,
)
from .notation import (
    DeviceType,
    Translated,
    device_name,
)


class _PrivateArrayMixin:

    def _constant_shape(self, node:ast.expr, what:str)->Optional[List[int]]:
        """The extents of a constant shape: an int, or a tuple of positive ints."""
        def one(candidate:ast.expr)->Optional[int]:
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, int) \
                    and candidate.value > 0:
                return candidate.value
            return None

        single:Optional[int] = one(node)
        if single is not None:
            return [single]
        if isinstance(node, ast.Tuple) and node.elts:
            extents:List[int] = []
            for element in node.elts:
                extent:Optional[int] = one(element)
                if extent is None:
                    return None
                extents.append(extent)
            return extents
        return None


    def _is_local_buffer(self, node:ast.expr)->bool:
        """True when ``node`` reaches `zeros_per_group(...)` memory, i.e. work-group shared storage.

        Both spellings count: the buffer itself (`bins`) and the group's slice of it
        (`bins[blockIdx]`). The slice form is what an atomic call passes -- `atomic_add(
        bins[blockIdx], i, 1)` -- and answering False there made the OpenCL cast say
        `__global`, which the compiler rejects as a change of address space.
        """
        if isinstance(node, ast.Subscript):
            node = node.value
        if not isinstance(node, ast.Name):
            return False
        t:DeviceType = self._lookup(node.id)
        return isinstance(t, tuple) and len(t) == 5 and t[0] == "shared"


    def _private_element_type(self, dtype_value:object)->Optional[Translated]:
        """(C element type, translator type) for a private array's element, or None."""
        from .. import structs

        if dtype_value is None:
            dtype_value = np.float64
        try:
            dtype:np.dtype = np.dtype(dtype_value)
        except TypeError:
            return None
        from ...vec_types.layout import _vector_dtype_name

        vector:Optional[str] = _vector_dtype_name(dtype)
        if vector is not None:
            # Device-side storage: the padded width is what the device type means here.
            return self._device_type(vector), vector, False
        if structs.is_record(dtype):
            name:str = structs.register(dtype)
            if dtype not in self.private_structs:
                self.private_structs.append(dtype)
            return name, name, False
        scalar:Optional[str] = DTYPE_MAP.get(dtype.name)
        if scalar is None:
            return None
        if scalar == "bool":
            scalar = "uchar"
        return self._device_type(scalar), scalar, True


    def _emit_private_array(self, name:str, node:ast.Assign)->bool:
        """Emit a per-work-item C array for `name = np.zeros(<constant shape>, ...)`.

        Returns False when the statement is not something a private array can represent, so
        the caller reports the reason.
        """
        call:ast.Call = node.value  # type: ignore[assignment]
        if isinstance(call.func, ast.Attribute):
            constructor:str = call.func.attr
        else:
            return False
        if constructor not in ("zeros", "ones", "full", "empty"):
            return False

        # np.zeros(4, ...) / np.zeros((2, 3), ...) -- one positional shape argument.
        positional:List[ast.expr] = list(call.args)
        try:
            shape_value:object = self._eval_constant(positional[0]) if positional else None
            dtype_node:Optional[ast.expr] = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "dtype"), None)
            dtype_value:object = self._eval_constant(dtype_node)
            fill_value:object = None
            if constructor == "full" and len(positional) > 1:
                fill_value = self._eval_constant(positional[1])
        except SyntaxError:
            return False

        if shape_value is None:
            shape:Tuple[int, ...] = ()
        elif isinstance(shape_value, int):
            shape = (shape_value,)
        elif isinstance(shape_value, (tuple, list)) and all(
                isinstance(extent, int) for extent in shape_value):
            shape = tuple(int(extent) for extent in shape_value)
        else:
            return False
        if not shape or any(extent <= 0 for extent in shape):
            return False

        element:Optional[Tuple[str, Any, bool]] = self._private_element_type(dtype_value)
        if element is None:
            return False
        ctype, element_t, scalar_element = element
        total:int = 1
        for extent in shape:
            total *= extent

        initialiser:Optional[str] = self._private_initialiser(constructor, total, fill_value,
                                                             scalar_element)
        if initialiser is None:
            return False
        self._emit(f"{ctype} {device_name(name)}[{total}]{initialiser};")
        self._define(name, ("bodyarr", total, shape, element_t), node)
        return True


    def _private_initialiser(self, constructor:str, total:int, fill_value:object,
                             scalar_element:bool)->Optional[str]:
        """The C initialiser for a private array, matching the numpy constructor.

        CUDA/HIP compile as C++, where a vector or structure aggregate is value-initialised
        with `{}`; OpenCL C needs `{0}`. A non-scalar element cannot take a `full`/`ones`
        value without a conversion the initialiser cannot express, so that is refused (the
        caller reports it) rather than emitting something that does not compile.
        """
        empty:str = " = {}" if self.backend != "opencl" else " = {0}"
        if constructor == "empty":
            return ""
        if constructor == "zeros":
            return empty
        value:object = 1 if constructor == "ones" else fill_value
        if value is None:
            return empty
        if not scalar_element:
            return None
        if total > self.PRIVATE_ARRAY_MAX_UNROLL:
            return None
        return " = {" + ", ".join(f"({value})" for _ in range(total)) + "}"
