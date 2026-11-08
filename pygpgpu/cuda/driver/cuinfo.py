"""The CUDA tables, assembled: the value types, the function signatures, the errors.

The tables live in the cuinfo_*.py modules and this class is the one name the drivers read.
The signature dicts are merged with ``{**a, **b}`` so the entry objects are shared: the
driver writes the resolved symbol into an entry, and the HIP side copies each CUDA entry
deliberately (a shared entry would hand one library another's function)."""
from __future__ import annotations

from typing import Any, Dict

from .cuinfo_types import (
    alter_types,
    basic_types,
    no_cached_info,
    no_need_check_error_funcs,
    scalar_types,
    vec_types,
)
from .cuinfo_errors import error_codes
from .cuinfo_sig_array import SIGNATURES as _SIG_ARRAY
from .cuinfo_sig_device import SIGNATURES as _SIG_DEVICE
from .cuinfo_sig_launch import SIGNATURES as _SIG_LAUNCH
from .cuinfo_sig_module import SIGNATURES as _SIG_MODULE


class CUInfo:
    """One namespace for every CUDA table the ctypes layer builds its bindings from."""

    basic_types = basic_types
    scalar_types = scalar_types
    vec_types = vec_types
    alter_types = alter_types
    no_need_check_error_funcs = no_need_check_error_funcs
    no_cached_info = no_cached_info
    error_codes = error_codes

    func_signatures:Dict[str, Any] = {
        **_SIG_DEVICE,
        **_SIG_ARRAY,
        **_SIG_MODULE,
        **_SIG_LAUNCH,
    }
