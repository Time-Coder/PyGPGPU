"""The typing hooks: the generic base over ParamSpec, and the launch records.

A user-defined generic over a ParamSpec needs Python 3.10+, and this package
supports 3.7+, so the base exists only for the type checkers; at runtime it is a
plain class whose __class_getitem__ returns itself, so subscripting
``PyKernel[P]`` never fails at import.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Generic, List, TypeVar
import numpy as np
from typing_extensions import ParamSpec, TypedDict

_P = ParamSpec("_P")
_R = TypeVar("_R")


if TYPE_CHECKING:
    class _KernelBase(Generic[_P]):
        """``Generic`` over the decorated function's parameters, for the type checkers."""
else:
    class _KernelBase:
        """Runtime stand-in: subscripting returns this class, so ``PyKernel[_P]`` imports."""
        def __class_getitem__(cls, item:object)->type:
            return cls


if TYPE_CHECKING:
    from ..translator import DeviceType, ReturnSiteLayout


# ---------- launch records ----------

# What `_bind_outer_values` hands the launch: the arguments carrying this call's outer-scope
# values, the declared outer scalars whose final value is written back, and the full
# argument list those values prefix.
class WritableOuter(TypedDict):
    name: str
    kind: str
    buffer: "np.ndarray[Any, Any]"
    type: "DeviceType"


class OuterState(TypedDict):
    args: List[object]
    writable: List[WritableOuter]
    full: List[object]


# The device-side slots a body's returns need: one buffer per value slot, plus the shared
# claim word (which work item won) and the site word (which `return` it was).
class ReturnState(TypedDict):
    buffers: List["np.ndarray[Any, Any]"]
    claim: "np.ndarray[Any, Any]"
    site: "np.ndarray[Any, Any]"


# What the translator records about the returns themselves: the dtype of each value slot,
# and the per-site layout that says how to rebuild the result.
class ReturnSlots(TypedDict):
    slot_types: List["np.dtype[Any]"]
    slot_kinds: List[str]
    arg_indices: Dict[str, int]
    sites: List["ReturnSiteLayout"]
