from .builtins import __all__ as _builtin_names
from .builtin_registry import BUILTIN_FUNCTIONS, builtin_function, builtin_name_of
from .builtins import *  # noqa: F403  (the dual-sided builtins, re-exported below)
from .decorator import (
    DEFAULT_BACKEND_ORDER,
    available_devices,
    get_default_device,
    get_device,
    get_type_map,
    kernel,
    list_devices,
    set_default_device,
    set_type_map,
    PyKernel,
)

__all__ = [
    "kernel", "PyKernel", "DEFAULT_BACKEND_ORDER",
    "list_devices", "get_device", "available_devices",
    "set_default_device", "get_default_device",
    "set_type_map", "get_type_map",
    # The builtin tables: the translator resolves a call through them, and tests use them
    # to check that the namespace and the device side agree.
    "BUILTIN_FUNCTIONS", "builtin_function", "builtin_name_of",
    *_builtin_names,
]
