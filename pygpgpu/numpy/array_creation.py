"""numpy's array-construction functions, returning gnp arrays.

Each wrapper runs numpy's own function and views the result as a gnp array. The results
own their memory (``np.empty``/``zeros``/``arange``/... all allocate), so nothing has to be
staged: the array a caller gets is the one the device will fill, and a kernel can read and
write it across any number of launches without it ever visiting the CPU.

Any gnp array passed *in* is brought back to the host first, because numpy cannot read
bytes a device is holding.
"""
from functools import wraps

import numpy as np

from .ndarray import ndarray

# Marks an array that owns its memory, so it needs no staging copy at all.
_OWNS_MEMORY = ("own",)


def _to_plain(value):
    """A value numpy can consume: gnp arrays become host-current numpy views."""
    if isinstance(value, ndarray):
        value._ensure_host_bytes()
        return value.view(np.ndarray)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_plain(item) for item in value)
    return value


def wrap_ndarray(func):
    """Run ``func`` on host-current inputs and view its numpy result as a gnp array."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*(_to_plain(arg) for arg in args),
                      **{key: _to_plain(value) for key, value in kwargs.items()})
        return _as_gnp(result)

    return wrapper


def _as_gnp(result):
    if isinstance(result, np.ndarray) and not isinstance(result, ndarray):
        return result.view(ndarray)
    if isinstance(result, tuple):
        return tuple(_as_gnp(item) for item in result)
    if isinstance(result, list):
        return [_as_gnp(item) for item in result]
    return result


# 1. From shape or value
empty = wrap_ndarray(np.empty)
zeros = wrap_ndarray(np.zeros)
ones = wrap_ndarray(np.ones)
full = wrap_ndarray(np.full)
empty_like = wrap_ndarray(np.empty_like)
zeros_like = wrap_ndarray(np.zeros_like)
ones_like = wrap_ndarray(np.ones_like)
full_like = wrap_ndarray(np.full_like)

# 2. From existing data
array = wrap_ndarray(np.array)
asarray = wrap_ndarray(np.asarray)
asanyarray = wrap_ndarray(np.asanyarray)
ascontiguousarray = wrap_ndarray(np.ascontiguousarray)
asfortranarray = wrap_ndarray(np.asfortranarray)
require = wrap_ndarray(np.require)

# 3. Numerical ranges
arange = wrap_ndarray(np.arange)
linspace = wrap_ndarray(np.linspace)
logspace = wrap_ndarray(np.logspace)
geomspace = wrap_ndarray(np.geomspace)

# 4. Building matrices
eye = wrap_ndarray(np.eye)
identity = wrap_ndarray(np.identity)
diag = wrap_ndarray(np.diag)
diagflat = wrap_ndarray(np.diagflat)
tri = wrap_ndarray(np.tri)
tril = wrap_ndarray(np.tril)
triu = wrap_ndarray(np.triu)
vander = wrap_ndarray(np.vander)

# 6. Miscellaneous
copy = wrap_ndarray(np.copy)
frombuffer = wrap_ndarray(np.frombuffer)
fromfile = wrap_ndarray(np.fromfile)
fromfunction = wrap_ndarray(np.fromfunction)
fromiter = wrap_ndarray(np.fromiter)
loadtxt = wrap_ndarray(np.loadtxt)
genfromtxt = wrap_ndarray(np.genfromtxt)
meshgrid = wrap_ndarray(np.meshgrid)

# 7. save
save = wrap_ndarray(np.save)
savez = wrap_ndarray(np.savez)
savez_compressed = wrap_ndarray(np.savez_compressed)
savetxt = wrap_ndarray(np.savetxt)
load = wrap_ndarray(np.load)