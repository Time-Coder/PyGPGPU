"""The binding machinery the host-side libraries share.

CUDA's math libraries and HIP's are bound the same way: a metaclass holds the DLL and
loads it lazily, a ``Func`` wrapper binds an entry point's argument and return types on
first use, and a returned status code becomes a ``LibraryError`` carrying the library's
own description of it.

The two backends differ in *what* they bind (their own DLL names, their own signature
tables, their own enumerations and status codes) and not in *how*, so the how lives here
and each backend's package brings its own tables.
"""
from __future__ import annotations

from ._base import (
    CUDA_ERROR_INVALID_CONTEXT,
    Library,
    LibraryError,
    MetaLibrary,
    add_dll_folder,
    resolve_dll,
)

__all__ = [
    "CUDA_ERROR_INVALID_CONTEXT",
    "Library",
    "LibraryError",
    "MetaLibrary",
    "add_dll_folder",
    "resolve_dll",
]
