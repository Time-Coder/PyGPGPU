"""``pygpgpu.numpy``: numpy-shaped arrays that stay on the device.

Everything numpy has is here -- the constructors return :class:`ndarray` objects that
behave like ``numpy.ndarray`` -- with one difference that matters for kernels: a gnp array
can keep its bytes on a device across any number of kernel calls. The bytes come back to
the host only when something actually needs them there.

::

    import pygpgpu.numpy as gnp
    from pygpgpu import kernel

    @kernel
    def saxpy(x, y, out, alpha):
        for i in range(out.shape[0]):
            out[i] = x[i] * alpha + y[i]

    a = gnp.arange(1_000_000, dtype=gnp.float32)   # host memory
    b = gnp.arange(1_000_000, dtype=gnp.float32)
    out = gnp.empty(1_000_000, dtype=gnp.float32)

    saxpy(a, b, out, 2.0)      # uploads once
    saxpy(a, b, out, 2.0)      # nothing moves: the arrays are already there
    print(out[0])              # ...and this is what brings them back

The explicit copy API is two asynchronous calls:

::

    a.to_device("cuda:0")     # -> Transfer; the copy runs on the GPU's stream
    a.to_host()               # -> Transfer; the next host read waits for it
    a.to_device("opencl:1").wait()   # block where the result is needed

``to_device`` also takes the kernel call's own selectors, and nothing else has to know
which backend was chosen. Between two devices -- and between two backends -- the copy is
staged through the host, because the device buffers are unrelated types.

``to_device(..., zero_copy=True)`` makes the device read and write the array's own host
memory. It is honoured on cuda, hip and metal, where the driver can guarantee that the
device addresses the very same bytes; on opencl the request raises instead of guessing,
because ``CL_MEM_USE_HOST_PTR`` does not promise the runtime keeps the caller's memory.

PyTorch tensors convert both ways, with the host-side forms able to share one buffer::

    gnp.ndarray.from_torch(tensor, zero_copy=True)
    array.to_torch(zero_copy=True)
"""
from numpy import *  # noqa: F403  (numpy's whole namespace, numpy's own spellings)

from . import _backends
from ._devices import Target, backend_devices, probe, resolve, target_of
from .array_creation import (
    empty, zeros, ones, full, empty_like, zeros_like, ones_like, full_like,
    array, asarray, asanyarray, ascontiguousarray, asfortranarray, require,
    arange, linspace, logspace, geomspace, eye, identity, diag, diagflat, tri,
    tril, triu, vander, copy, frombuffer, fromfile, fromfunction, fromiter,
    loadtxt, genfromtxt, meshgrid,
    save, savez, savez_compressed, savetxt, load
)
from .ndarray import ndarray
from .transfer import Transfer


def zero_copy_support(target: object = None) -> "tuple[bool, str]":
    """Whether that device can bind the array's own host memory, and why not.

    ``target`` takes anything ``to_device`` accepts; ``None`` means the default device. A
    bare backend name (``"cuda"``) is accepted too and answers for its first device, which
    is what makes the check usable before a device selector is known.

    Returns ``(supported, reason)``, so a caller can check before asking for it::

        ok, why = gnp.zero_copy_support("opencl")
        if not ok:
            print(why)     # names CL_MEM_USE_HOST_PTR and suggests zero_copy=False
    """
    from ._devices import BACKENDS, resolve

    selector = target
    if isinstance(selector, str):
        from ..kernel.decorator import _BACKEND_ALIASES

        backend = _BACKEND_ALIASES.get(selector.strip().lower())
        if backend is not None:
            if backend not in BACKENDS:
                # The interpreter has no device at all, so it cannot bind anything.
                return False, (
                    f"the {backend} backend runs in the interpreter and has no device "
                    f"memory, so zero-copy has no meaning there")
            # A bare backend answers from the backend itself: whether it *can* bind host
            # memory does not depend on this machine having one of its devices.
            return _backends.zero_copy_backend_support(backend)
    return _backends.zero_copy_supported(target_of(selector))


# ``to_host``/``to_device`` are methods; these aliases exist so a caller can import the
# names from the package and read the module's surface in one place.
to_host = ndarray.to_host
to_device = ndarray.to_device


__all__ = [
    "ndarray", "Transfer", "zero_copy_support", "to_host", "to_device",
    # device selection, in the kernel bracket's own spelling
    "Target", "resolve", "probe", "target_of", "backend_devices",
    # constructors (numpy's own names, re-exported)
    "empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like",
    "full_like", "array", "asarray", "asanyarray", "ascontiguousarray",
    "asfortranarray", "require", "arange", "linspace", "logspace", "geomspace",
    "eye", "identity", "diag", "diagflat", "tri", "tril", "triu", "vander", "copy",
    "frombuffer", "fromfile", "fromfunction", "fromiter", "loadtxt", "genfromtxt",
    "meshgrid", "save", "savez", "savez_compressed", "savetxt", "load",
]