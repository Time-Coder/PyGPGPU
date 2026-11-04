"""PyTorch tensor interop for gnp arrays.

Torch is an optional dependency: nothing here is imported until
``ndarray.from_torch``/``ndarray.to_torch`` is called, so ``import pygpgpu`` still needs
neither torch nor a GPU.

Two directions, and the same three rules on both:

* **Zero-copy needs shared storage, and that is only possible on the host.** A CPU tensor
  and a host-resident gnp array can be one buffer, so ``zero_copy=True`` hands the same
  memory to both sides. A CUDA tensor's storage is device memory owned by torch while a
  gnp array's bytes live in host memory, so "zero copy" between them is not a thing: the
  request raises rather than pretending.
* **A device-resident gnp array is brought home first.** Its host bytes are stale by
  design, so ``to_torch`` downloads before it builds anything.
* **Zero-copy sharing keeps both owners alive.** Whichever side created the storage holds
  the other, so the buffer cannot be freed while the other side still points at it.
"""
from __future__ import annotations

import weakref
from typing import Any

import numpy as np

from . import _backends

# torch's dtype spellings, keyed by numpy's kind+itemsize signature.
_TORCH_DTYPES: dict = {
    "b1": "bool",
    "i1": "int8",
    "i2": "int16",
    "i4": "int32",
    "i8": "int64",
    "u1": "uint8",
    "u2": "uint16",
    "u4": "uint32",
    "u8": "uint64",
    "f2": "float16",
    "f4": "float32",
    "f8": "float64",
    "c8": "complex64",
    "c16": "complex128",
}

_NUMPY_FROM_TORCH: dict = {
    "bool": np.bool_,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "complex64": np.complex64,
    "complex128": np.complex128,
}


def _torch() -> Any:
    """Import torch, or say what is missing."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "PyTorch interop needs torch: install it (pip install torch), or use the "
            "numpy copy instead (np.asarray(array))") from error
    return torch


def _torch_dtype_name(dtype: np.dtype) -> str:
    name = _TORCH_DTYPES.get(dtype.str[1:])
    if name is None:
        raise TypeError(
            f"torch has no dtype for a numpy '{dtype.name}' array; supported kinds are "
            f"{', '.join(sorted(_NUMPY_FROM_TORCH))}")
    return name


def _numpy_dtype_of(tensor: Any) -> np.dtype:
    spelling = str(tensor.dtype).replace("torch.", "")
    numpy_type = _NUMPY_FROM_TORCH.get(spelling)
    if numpy_type is None:
        raise TypeError(
            f"numpy has no dtype for a torch '{tensor.dtype}' tensor; convert it first")
    return np.dtype(numpy_type)


def _buffer_at(pointer: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """A writable numpy view over an existing buffer at ``pointer``.

    The memory belongs to someone else (a torch tensor), so nothing is allocated here:
    a ctypes array over the address gives numpy the view, and the caller keeps the owner
    alive for as long as the view exists.
    """
    import ctypes

    count = int(np.prod(shape)) if shape else 1
    byte_array = (ctypes.c_ubyte * (count * dtype.itemsize)).from_address(int(pointer))
    raw = np.frombuffer(byte_array, dtype=np.uint8)
    return raw.view(dtype).reshape(shape)


def _call_safely(function: Any) -> None:
    try:
        function()
    except BaseException:
        pass


def from_torch(cls: Any, tensor: Any, zero_copy: bool = False) -> Any:
    """Build a gnp array over (or copied from) a ``torch.Tensor``."""
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"from_torch() takes a torch.Tensor, got {type(tensor).__name__}")

    dtype = _numpy_dtype_of(tensor)
    shape = tuple(int(size) for size in tensor.shape)

    if tensor.is_cuda:
        if zero_copy:
            raise NotImplementedError(
                "a CUDA tensor's storage is device memory owned by torch while a gnp "
                "array's bytes live in host memory, so zero-copy between them is not "
                "possible. Use zero_copy=False (the default), which stages the values "
                "through the host")
        host = np.frombuffer(
            tensor.detach().to("cpu").contiguous().numpy().tobytes(), dtype=dtype)
        return cls(host.reshape(shape))

    if not tensor.is_contiguous():
        raise ValueError(
            "from_torch() needs a contiguous tensor (call tensor.contiguous() first); a "
            "strided tensor has no single buffer a kernel could index")

    count = int(tensor.numel())
    if count == 0:
        return cls(np.empty(shape, dtype=dtype))

    if not zero_copy:
        host = np.frombuffer(tensor.detach().numpy().tobytes(), dtype=dtype)
        return cls(host.reshape(shape))

    # Zero-copy: the array *is* the tensor's storage. Pin it so a device copy can run as
    # a real DMA over this memory, and keep the tensor alive for as long as the array.
    pointer = int(tensor.data_ptr())
    nbytes = count * dtype.itemsize
    unpin = _backends.page_locked(pointer, nbytes)
    array = cls(_buffer_at(pointer, shape, dtype))
    array._torch_source = tensor
    if unpin is not None:
        array._torch_unpin = unpin
        weakref.finalize(array, _call_safely, unpin)
    return array


def to_torch(array: Any, zero_copy: bool = False) -> Any:
    """Build a ``torch.Tensor`` over (or copied from) a gnp array."""
    torch = _torch()
    if not isinstance(array, np.ndarray):
        raise TypeError(f"to_torch() takes a numpy array, got {type(array).__name__}")

    # A device-resident array has stale host bytes; this pulls them home (and waits).
    array._ensure_host_bytes()
    plain = array.view(np.ndarray)
    shape = tuple(int(size) for size in plain.shape)

    if not zero_copy:
        # `from_numpy` shares, so the copy is made explicit here and the tensor owns it.
        return torch.from_numpy(np.array(plain, copy=True, order="C")).reshape(shape)

    if vars(array).get("_bindings"):
        # A device copy is either a separate buffer (whose host bytes may be stale) or a
        # host registration that already owns this memory for the GPU. Handing the array's
        # own bytes to torch in either case would let two owners write one buffer.
        raise NotImplementedError(
            "this array has device bindings, so its host memory is either stale or already "
            "shared with a device; release() them first, or use zero_copy=False")
    if not plain.flags["C_CONTIGUOUS"]:
        raise ValueError(
            "to_torch(zero_copy=True) needs a contiguous array; call "
            "np.ascontiguousarray first")

    tensor = torch.from_numpy(plain).reshape(shape)
    # The tensor points into the array's memory, so the array has to outlive it.
    array._torch_tensor = tensor
    return tensor