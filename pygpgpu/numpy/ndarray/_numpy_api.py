# The numpy surface: the ufunc dispatch, the torch bridge, and the plain-python
# readings of an array's values. The in-place operators are kept on the class,
# because numpy looks them up on the concrete type.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ndarray

from .plain import (
    _aliases,
    _call_ufunc,
    _to_plain_args,
    _to_plain_mapping,
)
import functools
from typing import Callable, Mapping, Optional, Tuple
import numpy as np
from numpy.typing import DTypeLike


def _inplace(func):
    @functools.wraps(func)
    def wrapper(self: "ndarray", *args: object, **kwargs: object) -> object:
        self._ensure_host_bytes()
        result = func(self, *args, **kwargs)
        self._after_host_write()
        return result
    return wrapper


@functools.lru_cache(maxsize=None)
def _element_class(dtype: np.dtype) -> Optional[type]:
    """The ctypes class a structured dtype was built from, or None for a plain record.

    numpy builds `np.dtype(pygf.float3)` from the vector class, and its own subscript answers
    an element as a live instance of that class (`arr[0]` is a `pygf.float3`, with the vector's
    arithmetic and components). gnp reads elements through the raw `ndarray.__getitem__` slot,
    which answers a plain `np.void` instead, so the class is recovered once per dtype and the
    element rebuilt from its bytes. numpy's own subscript returns a *snapshot* for a ctypes
    dtype -- a write through the element does not reach the array -- and this matches it.
    """
    if dtype.names is None:
        return None
    try:
        element = np.zeros(1, dtype=dtype)[0]
    except (TypeError, ValueError):
        return None
    return None if isinstance(element, np.void) else type(element)



class _NumpyApiMixin:

    @classmethod
    def from_torch(cls, tensor: object, zero_copy: bool = False) -> "ndarray":
        """A gnp array sharing (or holding a copy of) a ``torch.Tensor``'s values.

        A CPU tensor's storage is ordinary host memory, so with ``zero_copy=True`` the
        returned array *is* the tensor's buffer: writing through one is visible in the
        other. That needs the tensor to be contiguous and page-locked first, and the
        returned array keeps the pin alive for its lifetime. A CUDA tensor is uploaded
        once, and the two copies are then independent.
        """
        from .._torch import from_torch

        return from_torch(cls, tensor, zero_copy=zero_copy)


    def to_torch(self, zero_copy: bool = False) -> object:
        """A ``torch.Tensor`` holding this array's values.

        With ``zero_copy=True`` a host-side array hands its own buffer to torch, so the
        tensor and the array are one storage. A device-resident array has no host storage
        to lend, so it is copied to the host first and the tensor owns that copy.
        """
        from .._torch import to_torch

        return to_torch(self, zero_copy=zero_copy)


    def __array__(self, dtype: Optional[DTypeLike] = None,
                  copy: Optional[bool] = None) -> np.ndarray:
        self._ensure_host_bytes()
        plain = self.view(np.ndarray)
        if dtype is not None and plain.dtype != np.dtype(dtype):
            return plain.astype(dtype)
        return plain


    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: object,
                        **kwargs: object) -> object:
        self._ensure_host_bytes()
        result = _call_ufunc(ufunc, method, _to_plain_args(inputs),
                             _to_plain_mapping(kwargs))
        out = kwargs.get("out")
        if out is not None:
            # The class lives in the package facade, which imports this module, so the
            # runtime reference is made here (a TYPE_CHECKING import alone cannot answer
            # an isinstance, and a top-level one would close the import cycle).
            from . import ndarray
            for value in (out if isinstance(out, tuple) else (out,)):
                if isinstance(value, ndarray):
                    value._after_host_write()
        return self._change_result(result)


    def __array_function__(self, func: Callable[..., object],
                           types: Tuple[type, ...], args: Tuple[object, ...],
                           kwargs: Mapping[str, object]) -> object:
        self._ensure_host_bytes()
        result = func(*_to_plain_args(args), **_to_plain_mapping(kwargs))
        return self._change_result(result)


    def _change_to_ndarray(self, value: object) -> object:
        # The class lives in the package facade, which imports this module: the import is
        # made where it is used, so the two do not import each other at module level.
        from . import ndarray
        if value is self:
            return value
        if not isinstance(value, ndarray):
            value = value.view(ndarray)
        value._bindings = {}
        value._family = None
        value._staging = None
        value._staging_epoch = 0
        value._arg_name = ""
        value._should_copy = True
        value._children = []
        if _aliases(value, self):
            value._parent = self
            self._children.append(value)
        else:
            value._parent = None
        return value


    def _change_result(self, result: object) -> object:
        if isinstance(result, np.ndarray):
            return self._change_to_ndarray(result)
        if isinstance(result, tuple):
            return tuple(self._change_to_ndarray(item) if isinstance(item, np.ndarray)
                         else item for item in result)
        return result


    def __getitem__(self, key: object) -> object:
        self._ensure_host_bytes()
        element = np.ndarray.__getitem__(self.view(np.ndarray), key)
        # A ctypes-backed dtype answers its element as an instance of the class itself on
        # numpy's own subscript path; the raw slot above answers a plain void, so rebuild the
        # snapshot numpy would have returned (a copy -- writes through it do not reach the
        # array, exactly as numpy's element read behaves).
        scalar_type: Optional[type] = _element_class(self.dtype)
        if scalar_type is not None and isinstance(element, np.void):
            try:
                element = scalar_type.from_buffer_copy(element.tobytes())
            except (TypeError, ValueError):
                pass
        return self._change_result(element)


    def __setitem__(self, key: object, value: object) -> None:
        self._ensure_host_bytes()
        # The class lives in the package facade, which imports this module: the import is
        # made where it is used, so the two do not import each other at module level.
        from . import ndarray
        if isinstance(value, ndarray):
            value._ensure_host_bytes()
            value = value.view(np.ndarray)
        scalar_type: Optional[type] = _element_class(self.dtype)
        if scalar_type is not None and isinstance(value, scalar_type):
            # numpy's own store path accepts a ctypes instance; the raw slot reads it as a
            # sequence and refuses, so hand over the dtype's own void scalar instead.
            value = np.frombuffer(bytes(value), dtype=self.dtype)[0]
        np.ndarray.__setitem__(self.view(np.ndarray), key, value)
        self._after_host_write()


    def __repr__(self) -> str:
        self._ensure_host_bytes()
        return np.ndarray.__repr__(self.view(np.ndarray))


    def __str__(self) -> str:
        self._ensure_host_bytes()
        return np.ndarray.__str__(self.view(np.ndarray))


    def tolist(self) -> object:
        self._ensure_host_bytes()
        return np.ndarray.tolist(self.view(np.ndarray))


    def tobytes(self) -> bytes:
        self._ensure_host_bytes()
        return np.ndarray.tobytes(self.view(np.ndarray))


    def tofile(self, fid: object, sep: str = "", format: str = "%s") -> None:
        self._ensure_host_bytes()
        np.ndarray.tofile(self.view(np.ndarray), fid, sep, format)


    def copy(self, order: str = "C") -> "ndarray":
        self._ensure_host_bytes()
        return self._change_result(np.ndarray.copy(self.view(np.ndarray), order))


    def dumps(self) -> bytes:
        self._ensure_host_bytes()
        return np.ndarray.dumps(self.view(np.ndarray))


    def dump(self, file: object) -> None:
        self._ensure_host_bytes()
        np.ndarray.dump(self.view(np.ndarray), file)


    def item(self, *args: int) -> object:
        self._ensure_host_bytes()
        return np.ndarray.item(self.view(np.ndarray), *args)


    def itemset(self, *args: object) -> None:
        self._ensure_host_bytes()
        np.ndarray.itemset(self.view(np.ndarray), *args)
        self._after_host_write()


    @property
    def T(self) -> "ndarray":
        self._ensure_host_bytes()
        return self._change_to_ndarray(np.transpose(self.view(np.ndarray)))


    @property
    def data(self) -> memoryview:
        self._ensure_host_bytes()
        return self.view(np.ndarray).data


    @property
    def real(self) -> "ndarray":
        self._ensure_host_bytes()
        return self._change_result(self.view(np.ndarray).real)


    @property
    def imag(self) -> "ndarray":
        self._ensure_host_bytes()
        return self._change_result(self.view(np.ndarray).imag)


    @property
    def flat(self) -> np.flatiter:
        self._ensure_host_bytes()
        return self.view(np.ndarray).flat


    @property
    def ctypes(self) -> object:
        self._ensure_host_bytes()
        return self.view(np.ndarray).ctypes


    @property
    def base(self) -> Optional["ndarray"]:
        return self.__dict__.get("_parent")
