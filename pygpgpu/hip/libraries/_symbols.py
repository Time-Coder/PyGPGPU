"""The HIP spelling of a CUDA library symbol.

hipBLAS is the vendor's port of cuBLAS: the API is the same, the argument lists are the
same, and only the names differ. The mapping is not a plain prefix swap, though:

* the prefix changes (``cublas`` -> ``hipblas``, ``cublasLt`` -> ``hipblasLt``);
* **hipBLAS drops the ABI version suffix** that CUDA 12 carries -- ``cublasSgemm_v2`` is
  ``hipblasSgemm`` there, because HIP has no second ABI;
* hipBLASLt's entry points are spelled ``hipblasLt*`` (camel case) even though its DLL is
  ``hipblaslt.dll``;
* a few entries have no counterpart at all: ``GetStatusString`` is absent, and hipBLAS has
  no version query.

So the mapping is **derived from the CUDA signature table** rather than written out
again: the HIP driver declares which CUDA library it mirrors, and the names follow from
that plus the exceptions below. A change to the CUDA side cannot leave the HIP side
behind.
"""
from __future__ import annotations

from typing import Dict, Optional

__all__ = ["hip_symbol", "MIRRORED_LIBRARIES", "UNMIRRORED_SYMBOLS"]


# The CUDA library each HIP library mirrors. The key is the HIP library's own name.
MIRRORED_LIBRARIES:Dict[str, str] = {
    "hipblas": "cublas",
    "hipblaslt": "cublasLt",
}


# The prefix each HIP library's SYMBOLS carry, which is not always the library's own
# name: hipBLASLt's DLL is `hipblaslt.dll` but its entry points are `hipblasLt*`, the
# camel-case spelling CUDA's cuBLASLt uses.
_SYMBOL_PREFIX:Dict[str, str] = {
    "hipblas": "hipblas",
    "hipblaslt": "hipblasLt",
}


# The CUDA symbols with no HIP counterpart, by the HIP library that lacks them. A CUDA
# symbol that is absent here is expected to exist on the HIP side under the mapped name;
# one that is listed is not bound there at all.
UNMIRRORED_SYMBOLS:Dict[str, set] = {
    "hipblas": {"cublasGetStatusString", "cublasGetVersion", "cublasGetVersion_v2"},
    "hipblaslt": {"cublasLtGetStatusString"},
}


# The CUDA libraries whose HIP counterpart drops the ABI version suffix. hipBLAS is the
# only one: hipBLASLt keeps CUDA's spelling verbatim.
_DROP_ABI_SUFFIX:Dict[str, bool] = {
    "hipblas": True,
    "hipblaslt": False,
}


def hip_symbol(hip_library:str, cuda_symbol:str)->Optional[str]:
    """The HIP spelling of a CUDA symbol, or None when the HIP library has none.

    The CUDA symbol's own prefix decides the replacement, and the replacement is the
    HIP library's SYMBOL prefix (which is not always its own name) plus the rest of the
    name -- with the ABI version suffix dropped for the libraries that have no second
    ABI.
    """
    cuda_library:str = MIRRORED_LIBRARIES.get(hip_library, "")
    if not cuda_library:
        return None
    if cuda_symbol in UNMIRRORED_SYMBOLS.get(hip_library, set()):
        return None
    if not cuda_symbol.startswith(cuda_library):
        return None
    tail:str = cuda_symbol[len(cuda_library):]
    if _DROP_ABI_SUFFIX.get(hip_library, False) and tail.endswith("_v2"):
        tail = tail[:-3]
    return _SYMBOL_PREFIX.get(hip_library, hip_library) + tail
