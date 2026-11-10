# A creation below the magic loop: a private C array per work item.
from __future__ import annotations
import ast
import types
from typing import (Any, List, Optional, Tuple)
import numpy as np
from .nd_index import (
    _constant_extent,
)
from .type_map import (
    DTYPE_MAP,
)
from .notation import (
    BLOCK_ARRAY,
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


    def _baked_shape(self, node:ast.expr)->Optional[List[int]]:
        """The extents of a shape read from a *value* this launch carries.

        A shared buffer's size is baked into the generated source (`__shared__ T name[N]` is
        a fixed-size declaration), so when the shape names a kernel parameter the parameter's
        value specializes the kernel: the number is read here, once, and the generated source
        carries the literal. A different value compiles a kernel of its own, because the value
        is folded into the signature digest that hashes the C name (see `_source_signature`).
        A module constant bakes the same way -- it is one number for the life of the process,
        the same rule `np.ndindex(HEIGHT, WIDTH)` already follows.

        A parameter whose value is not a usable size is a different refusal from "the name
        cannot be resolved": the user did everything the feature asks, and the *value* this
        launch carried is the problem, so the message names the value instead of telling them
        to use a parameter.
        """
        def bake(candidate:ast.expr)->Optional[int]:
            if isinstance(candidate, ast.Name):
                # A kernel parameter: this launch's value, positional order.
                position:Optional[int] = next(
                    (position for position, (name, _) in enumerate(self.params)
                     if name == candidate.id), None)
                if position is not None and position < len(self.argument_values):
                    value:object = self.argument_values[position]
                    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                        self._error(
                            f"a work-group array's shape read from parameter '{candidate.id}' "
                            f"must be a whole number of elements, and this launch carried "
                            f"{value!r}: the size is baked into the compiled kernel, so it "
                            f"cannot be a fraction or a switch", candidate)
                    if int(value) <= 0:
                        self._error(
                            f"a work-group array's shape read from parameter '{candidate.id}' "
                            f"must be positive, and this launch carried {int(value)}: a "
                            f"shared buffer with no elements has nothing to index",
                            candidate)
                    return int(value)
                # A module-level constant (or an attribute of one): one number for the
                # process, which is what `_constant_extent` already bakes for an
                # `np.ndindex` bound.
                resolved:object = self.resolve_global(candidate.id)
                extent:Optional[int] = _constant_extent(resolved)
                if extent is None and resolved is not None and not isinstance(
                        resolved, (int, np.integer)):
                    self._error(
                        f"a work-group array's shape read from module constant "
                        f"'{candidate.id}' must be a whole number of elements, and it is "
                        f"{resolved!r}", candidate)
                return extent
            if isinstance(candidate, ast.Attribute) and isinstance(candidate.value, ast.Name):
                holder:object = self.resolve_global(candidate.value.id)
                if isinstance(holder, types.ModuleType):
                    return _constant_extent(getattr(holder, candidate.attr, None))
            return None

        single:Optional[int] = bake(node)
        if single is not None:
            return [single]
        if isinstance(node, ast.Tuple) and node.elts:
            extents:List[int] = []
            for element in node.elts:
                extent:Optional[int] = bake(element)
                if extent is None:
                    return None
                extents.append(extent)
            return extents
        return None


    def _is_local_buffer(self, node:ast.expr)->bool:
        """True when ``node`` reaches work-group shared storage, i.e. `__local` on OpenCL.

        Both kinds count: a per-group buffer with its group subscript
        (`bins[blockIdx]`) and a block-local one (`bins`). The spelling an atomic call passes
        is what decides here -- `atomic_add(bins[blockIdx], i, 1)` and
        `atomic_add(bins, i, 1)` are both work-group memory -- and answering False made the
        OpenCL cast say `__global`, which the compiler rejects as a change of address space.
        """
        if isinstance(node, ast.Subscript):
            node = node.value
        if not isinstance(node, ast.Name):
            return False
        t:DeviceType = self._lookup(node.id)
        return (isinstance(t, tuple) and len(t) == 5
                and t[0] in ("shared", BLOCK_ARRAY))


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
