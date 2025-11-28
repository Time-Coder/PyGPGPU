from functools import wraps

import numpy as np

from .ndarray import ndarray


def wrap_ndarray(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        for value in args:
            if isinstance(value, ndarray):
                value.to_host()

        for value in kwargs.values():
            if isinstance(value, ndarray):
                value.to_host()

        result = func(*args, **kwargs)
        if isinstance(result, np.ndarray) and not isinstance(result, ndarray):
            result = ndarray(result)

        if isinstance(result, tuple):
            result = list(result)
            for i in range(len(result)):
                if isinstance(result[i], np.ndarray) and not isinstance(result[i], ndarray):
                    result[i] = ndarray(result[i])

            result = tuple(result)

        return result
    
    return wrapper


# 1. From shape or value
empty = wrap_ndarray(np.empty)
zeros = wrap_ndarray(np.zeros)
ones = wrap_ndarray(np.ones)
full = wrap_ndarray(np.full)

# # 2. From existing data
array = wrap_ndarray(np.array)
asarray = wrap_ndarray(np.asarray)
asanyarray = wrap_ndarray(np.asanyarray)
ascontiguousarray = wrap_ndarray(np.ascontiguousarray)
asfortranarray = wrap_ndarray(np.asfortranarray)
require = wrap_ndarray(np.require)

# # 3. Numerical ranges
arange = wrap_ndarray(np.arange)
linspace = wrap_ndarray(np.linspace)
logspace = wrap_ndarray(np.logspace)
geomspace = wrap_ndarray(np.geomspace)

# # 4. Building matrices
eye = wrap_ndarray(np.eye)
identity = wrap_ndarray(np.identity)
diag = wrap_ndarray(np.diag)
diagflat = wrap_ndarray(np.diagflat)
tri = wrap_ndarray(np.tri)
tril = wrap_ndarray(np.tril)
triu = wrap_ndarray(np.triu)
vander = wrap_ndarray(np.vander)

# # 6. Miscellaneous
copy = wrap_ndarray(np.copy)
frombuffer = wrap_ndarray(np.frombuffer)
fromfile = wrap_ndarray(np.fromfile)
fromfunction = wrap_ndarray(np.fromfunction)
fromiter = wrap_ndarray(np.fromiter)
fromstring = wrap_ndarray(np.fromstring)
loadtxt = wrap_ndarray(np.loadtxt)
genfromtxt = wrap_ndarray(np.genfromtxt)
meshgrid = wrap_ndarray(np.meshgrid)

# # 7. save
save = wrap_ndarray(np.save)
savez = wrap_ndarray(np.savez)
savez_compressed = wrap_ndarray(np.savez_compressed)
savetxt = wrap_ndarray(np.savetxt)
load = wrap_ndarray(np.load)
