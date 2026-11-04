"""The binding machinery, from the shared package.

The CUDA math libraries and the HIP ones are bound the same way -- lazily loaded DLL, a
signature table, a status code turned into a ``LibraryError`` -- so the machinery lives in
``pygpgpu/libraries`` and each backend's package brings its own tables. This module is a
re-export, kept so the libraries of this package import ``._base`` as before.
"""
from __future__ import annotations

from ...libraries._base import (
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
