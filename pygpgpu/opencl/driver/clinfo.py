"""The CL tables, assembled: the value types, the function signatures, the get-info maps.

The tables live in the clinfo_*.py modules and this class is the one name the drivers and the
import hooks read. The signature dicts are merged with ``{**a, **b}`` rather than copied
entry by entry, because the driver writes the resolved symbol into an entry
(``func_info["dll_func"] = func``) and the HIP side copies each CUDA entry deliberately: a
second copy of an entry here would hand one library another's function."""
from __future__ import annotations

from typing import Any, Dict

from .clinfo_types import (
    alter_types,
    basic_types,
    image_types,
    scalar_types,
    vec_types,
)
from .clinfo_query_types import (
    command_queue_info_types,
    context_info_types,
    device_info_types,
    event_info_types,
    kernel_info_types,
    mem_info_types,
    no_cached_info,
    platform_info_types,
    program_build_info_types,
    program_info_types,
    sampler_info_types,
)
from .clinfo_sig_context import SIGNATURES as _SIG_CONTEXT
from .clinfo_sig_device import SIGNATURES as _SIG_DEVICE
from .clinfo_sig_enqueue import SIGNATURES as _SIG_ENQUEUE
from .clinfo_sig_event import SIGNATURES as _SIG_EVENT
from .clinfo_sig_memobj import SIGNATURES as _SIG_MEMOBJ
from .clinfo_sig_misc import SIGNATURES as _SIG_MISC
from .clinfo_sig_platform import SIGNATURES as _SIG_PLATFORM
from .clinfo_sig_program import SIGNATURES as _SIG_PROGRAM


class CLInfo:
    """One namespace for every OpenCL table the ctypes layer builds its bindings from."""

    image_types = image_types
    basic_types = basic_types
    scalar_types = scalar_types
    vec_types = vec_types
    alter_types = alter_types

    # Each entry maps a CL function name to a heterogeneous record: "args" is a dict of
    # per-argument info, "errors" maps an error code to its message, "restype"/"dll_func"
    # hold the return type and the resolved symbol. Annotated Dict[str, Any] because
    # inferred literal types do not include every key combination the callers index.
    func_signatures:Dict[str, Any] = {
        **_SIG_PLATFORM,
        **_SIG_DEVICE,
        **_SIG_CONTEXT,
        **_SIG_PROGRAM,
        **_SIG_MEMOBJ,
        **_SIG_ENQUEUE,
        **_SIG_EVENT,
        **_SIG_MISC,
    }

    platform_info_types = platform_info_types
    device_info_types = device_info_types
    context_info_types = context_info_types
    program_info_types = program_info_types
    program_build_info_types = program_build_info_types
    kernel_info_types = kernel_info_types
    command_queue_info_types = command_queue_info_types
    mem_info_types = mem_info_types
    event_info_types = event_info_types
    sampler_info_types = sampler_info_types
    no_cached_info = no_cached_info
