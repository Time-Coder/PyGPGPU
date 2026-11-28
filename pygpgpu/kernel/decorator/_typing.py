"""The typing hooks: the generic base over ParamSpec, and the launch records.

A user-defined generic over a ParamSpec needs Python 3.10+, and this package
supports 3.7+, so the base exists only for the type checkers; at runtime it is a
plain class whose __class_getitem__ returns itself, so subscripting
``PyKernel[P]`` never fails at import.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, TypeVar
import numpy as np
from typing_extensions import ParamSpec, TypedDict

_P = ParamSpec("_P")
_R = TypeVar("_R")


if TYPE_CHECKING:
    class _KernelBase(Generic[_P]):
        """``Generic`` over the decorated function's parameters, for the type checkers."""
else:
    class _KernelBase:
        """Runtime stand-in: subscripting returns this class, so ``PyKernel[_P]`` imports.

        It also carries the four AD properties, because this is the base the kernel object
        and both wrappers share: `obj.trace.grad(...)` has to work or a differentiable
        method cannot be differentiated from an instance, and a method body is a common
        shape. Each property imports `pygpgpu.ad` on first access -- `import pygpgpu` must
        not pull the AST transform in -- and a `k.grad(...)` form is a different object
        from an array's `.grad` buffer: different objects, different call shapes, and the
        docstrings settle which is which.
        """
        def __class_getitem__(cls, item:object)->type:
            return cls

        def _ad_subject(self)->Tuple[object, object, Optional[str]]:
            """The kernel object a form is built over, and the instance it binds."""
            kernel:object = getattr(self, "_kernel", self)
            instance:object = getattr(self, "_instance", None)
            backend:Optional[str] = getattr(self, "_backend", None)
            return kernel, instance, backend

        @property
        def grad(self)->object:
            """The gradient form: `k.grad(loss=...)(args)` answers gradients only."""
            from ...ad import GradForm
            kernel, instance, backend = self._ad_subject()
            return GradForm(kernel, instance=instance, backend=backend)

        @property
        def value_and_grad(self)->object:
            """The gradient form that also answers the loss the forward computed."""
            from ...ad import ValueAndGradForm
            kernel, instance, backend = self._ad_subject()
            return ValueAndGradForm(kernel, instance=instance, backend=backend)

        @property
        def vjp(self)->object:
            """The pullback form: `result, pullback = k.vjp()(args)`, then `pullback(g)`."""
            from ...ad import VJPForm
            kernel, instance, backend = self._ad_subject()
            return VJPForm(kernel, instance=instance, backend=backend)

        @property
        def jvp(self)->object:
            """The forward-mode form: `dout = k.jvp(tangent=dx)(args)`."""
            from ...ad import JVPForm
            kernel, instance, backend = self._ad_subject()
            return JVPForm(kernel, instance=instance, backend=backend)


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


class OuterObjectBinding(TypedDict):
    """An outer python object that travelled as a structure and may be written back."""
    name: str
    source: object
    instance: object
    written: List[str]


class OuterState(TypedDict):
    args: List[object]
    writable: List[WritableOuter]
    full: List[object]
    objects: List[OuterObjectBinding]


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
