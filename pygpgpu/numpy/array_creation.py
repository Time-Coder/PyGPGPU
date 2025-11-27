from typing import Union, Tuple

import numpy as np

from .ndarray import ndarray




# 1. From shape or value
def empty(shape, dtype=float, order='C') -> ndarray:
    return ndarray(np.empty(shape, dtype=dtype, order=order))

def empty_like(prototype, dtype=None, order='K', subok=True, shape=None) -> ndarray:
    return ndarray(np.empty_like(prototype, dtype=dtype, order=order, subok=False, shape=shape))

def zeros(shape, dtype=float, order='C') -> ndarray:
    return ndarray(np.zeros(shape, dtype=dtype, order=order))

def zeros_like(prototype, dtype=None, order='K', subok=True, shape=None) -> ndarray:
    return ndarray(np.zeros_like(prototype, dtype=dtype, order=order, subok=False, shape=shape))

def ones(shape, dtype=float, order='C') -> ndarray:
    return ndarray(np.ones(shape, dtype=dtype, order=order))

def ones_like(prototype, dtype=None, order='K', subok=True, shape=None) -> ndarray:
    return ndarray(np.ones_like(prototype, dtype=dtype, order=order, subok=False, shape=shape))

def full(shape, fill_value, dtype=None, order='C') -> ndarray:
    return ndarray(np.full(shape, fill_value, dtype=dtype, order=order))

def full_like(prototype, fill_value, dtype=None, order='K', subok=True, shape=None) -> ndarray:
    return ndarray(np.full_like(prototype, fill_value, dtype=dtype, order=order, subok=False, shape=shape))


# 2. From existing data
def array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None) -> ndarray:
    if like is not None:
        raise NotImplementedError("like parameter not supported in ndarray.array")
    
    arr = np.array(object, dtype=dtype, copy=copy, order=order, subok=False, ndmin=ndmin)
    return ndarray(arr)

def asarray(a, dtype=None, order=None, *, like=None) -> ndarray:
    if like is not None:
        raise NotImplementedError("like parameter not supported")
    
    return ndarray(np.asarray(a, dtype=dtype, order=order))

def asanyarray(a, dtype=None, order=None, *, like=None) -> ndarray:
    if like is not None:
        raise NotImplementedError("like parameter not supported")
    
    return ndarray(np.asarray(a, dtype=dtype, order=order))

def ascontiguousarray(a, dtype=None, *, like=None) -> ndarray:
    if like is not None:
        raise NotImplementedError("like parameter not supported")
    
    return ndarray(np.ascontiguousarray(a, dtype=dtype))

def asfortranarray(a, dtype=None, *, like=None) -> ndarray:
    if like is not None:
        raise NotImplementedError("like parameter not supported")
    
    return ndarray(np.asfortranarray(a, dtype=dtype))

def require(a, dtype=None, requirements=None) -> ndarray:
    return ndarray(np.require(a, dtype=dtype, requirements=requirements))


# 3. Numerical ranges
def arange(start, stop=None, step=None, dtype=None, *, like=None) -> ndarray:
    if like is not None:
        raise NotImplementedError("like parameter not supported")
    
    return ndarray(np.arange(start, stop, step, dtype=dtype))

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0) -> Union[ndarray, Tuple[ndarray, float]]:
    result = np.linspace(start, stop, num=num, endpoint=endpoint,
                         retstep=retstep, dtype=dtype, axis=axis)
    if retstep:
        arr, step = result
        return ndarray(arr), step
    else:
        return ndarray(result)

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0) -> ndarray:
    return ndarray(np.logspace(start, stop, num=num, endpoint=endpoint,
                                  base=base, dtype=dtype, axis=axis))

def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0) -> ndarray:
    return ndarray(np.geomspace(start, stop, num=num, endpoint=endpoint,
                                   dtype=dtype, axis=axis))


# 4. Building matrices
def eye(N, M=None, k=0, dtype=float, order='C') -> ndarray:
    return ndarray(np.eye(N, M=M, k=k, dtype=dtype, order=order))

def identity(n, dtype=float) -> ndarray:
    return ndarray(np.identity(n, dtype=dtype))

def diag(v, k=0) -> ndarray:
    return ndarray(np.diag(v, k=k))

def diagflat(v, k=0) -> ndarray:
    return ndarray(np.diagflat(v, k=k))

def tri(N, M=None, k=0, dtype=float) -> ndarray:
    return ndarray(np.tri(N, M=M, k=k, dtype=dtype))

def tril(m, k=0) -> ndarray:
    return ndarray(np.tril(m, k=k))

def triu(m, k=0) -> ndarray:
    return ndarray(np.triu(m, k=k))

def vander(x, N=None, increasing=False) -> ndarray:
    return ndarray(np.vander(x, N=N, increasing=increasing))


# 5. The Matrix class (deprecated, but included for completeness)
# Note: np.mat is deprecated; we do not wrap it.


# 6. Miscellaneous
def copy(a, order='K', subok=False) -> ndarray:
    return ndarray(np.copy(a, order=order, subok=subok))

def frombuffer(buffer, dtype=float, count=-1, offset=0) -> ndarray:
    return ndarray(np.frombuffer(buffer, dtype=dtype, count=count, offset=offset))

def fromfile(file, dtype=float, count=-1, sep='', offset=0, *, like=None) -> ndarray:
    if like is not None:
        raise NotImplementedError("like parameter not supported")
    
    return ndarray(np.fromfile(file, dtype=dtype, count=count, sep=sep, offset=offset))

def fromfunction(function, shape, *, dtype=float, like=None, **kwargs) -> ndarray:
    if like is not None:
        raise NotImplementedError("like parameter not supported")
    
    return ndarray(np.fromfunction(function, shape, dtype=dtype, **kwargs))

def fromiter(iter, dtype, count=-1, *, like=None) -> ndarray:
    if like is not None:
        raise NotImplementedError("like parameter not supported")
    
    return ndarray(np.fromiter(iter, dtype=dtype, count=count))

def fromstring(string, dtype=float, count=-1, *, sep, like=None) -> ndarray:
    if like is not None:
        raise NotImplementedError("like parameter not supported")
    
    return ndarray(np.fromstring(string, dtype=dtype, count=count, sep=sep))

def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes',
            max_rows=None, *, like=None) -> Union[ndarray, Tuple[ndarray, ...]]:
    if like is not None:
        raise NotImplementedError("like parameter not supported")
    
    result = np.loadtxt(fname, dtype=dtype, comments=comments, delimiter=delimiter,
                        converters=converters, skiprows=skiprows, usecols=usecols,
                        unpack=unpack, ndmin=ndmin, encoding=encoding, max_rows=max_rows)
    if unpack and isinstance(result, tuple):
        return tuple(ndarray(arr) for arr in result)
    else:
        return ndarray(result)

def genfromtxt(fname, dtype=float, comments='#', delimiter=None, skip_header=0,
               skip_footer=0, converters=None, missing_values=None, filling_values=None,
               usecols=None, names=None, excludelist=None, deletechars=None,
               replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i',
               unpack=None, usemask=False, loose=True, invalid_raise=True,
               max_rows=None, encoding=None, *, like=None) -> ndarray:
    if like is not None:
        raise NotImplementedError("like parameter not supported")
    
    return ndarray(np.genfromtxt(fname, dtype=dtype, comments=comments, delimiter=delimiter,
                                    skip_header=skip_header, skip_footer=skip_footer,
                                    converters=converters, missing_values=missing_values,
                                    filling_values=filling_values, usecols=usecols,
                                    names=names, excludelist=excludelist, deletechars=deletechars,
                                    replace_space=replace_space, autostrip=autostrip,
                                    case_sensitive=case_sensitive, defaultfmt=defaultfmt,
                                    unpack=unpack, usemask=usemask, loose=loose,
                                    invalid_raise=invalid_raise, max_rows=max_rows,
                                    encoding=encoding))

def meshgrid(*xi, indexing='xy', sparse=False, copy=True) -> Tuple[ndarray, ...]:
    arrays = np.meshgrid(*xi, indexing=indexing, sparse=sparse, copy=copy)
    return tuple(ndarray(arr) for arr in arrays)

# 7. save
def save(file, arr, allow_pickle:bool=True):
    if isinstance(arr, ndarray):
        arr.to_host()
        
    np.save(file, arr, allow_pickle)

def savez(file, *args, allow_pickle:bool=True, **kwds):
    for arg in args:
        if isinstance(arg, ndarray):
            arg.to_host()

    for value in kwds.values():
        if isinstance(value, ndarray):
            value.to_host()

    np.savez(file, *args, allow_pickle, **kwds)

def savez_compressed(file, *args, allow_pickle=True, **kwds):
    for arg in args:
        if isinstance(arg, ndarray):
            arg.to_host()

    for value in kwds.values():
        if isinstance(value, ndarray):
            value.to_host()

    np.savez_compressed(file, *args, allow_pickle, **kwds)

def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None):
    if isinstance(X, ndarray):
        X.to_host()

    np.savetxt(fname, X, fmt, delimiter, newline, header, footer, comments, encoding)

def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII', *, max_header_size=10000):
    result = np.load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size=max_header_size)
    if isinstance(result, np.ndarray) and not isinstance(result, ndarray):
        result = ndarray(result)

    if isinstance(result, dict):
        keys = list(result.keys())
        for key in keys:
            value = result[key]
            if isinstance(value, np.ndarray) and not isinstance(value, ndarray):
                result[key] = ndarray(value)

    return result