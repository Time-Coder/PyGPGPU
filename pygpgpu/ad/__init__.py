# The public surface of the AD stage: the forms, the tape, the checker.
#
# This module holds the names and nothing else -- the transformation logic lives in
# forward.py / reverse.py, the per-op rules in rules/, the launch bookkeeping in tape.py.
# It is imported lazily, on first touch of a kernel's `.grad` / `.vjp` / `.jvp` property
# or of `Tape`, so a process that never differentiates never pays for it.
from __future__ import annotations

from .check_grad import check_grad
from .forms import GradForm, ValueAndGradForm, VJPForm
from .forward_form import JVPForm
from .grad_for import grad_for
from .tape import Tape

__all__ = [
    "grad", "value_and_grad", "vjp", "jvp", "check_grad", "Tape", "grad_for",
    "GradForm", "ValueAndGradForm", "VJPForm", "JVPForm",
]


def grad(kernel:object, **config:object)->GradForm:
    """The gradient form of a kernel: `pygpgpu.ad.grad(k, wrt=..., loss=...)(args)`.

    The same transform the `k.grad(...)` property reaches; spelled here so the entry
    points are importable where there is no kernel object to hang a property on. A
    *function that is several kernels* is `Tape`'s subject, not this one: a sequence of
    launches with host code between them records what ran rather than transforming
    something known ahead of time.
    """
    from ..kernel.decorator import PyKernel
    if not isinstance(kernel, PyKernel):
        raise TypeError(
            "ad.grad differentiates one @kernel function: for a sequence of launches "
            "with host code between them, use pygpgpu.Tape")
    return GradForm(kernel, **config)


def value_and_grad(kernel:object, **config:object)->ValueAndGradForm:
    """The gradient form that also answers the loss value the forward computed."""
    from ..kernel.decorator import PyKernel
    if not isinstance(kernel, PyKernel):
        raise TypeError(
            "ad.value_and_grad differentiates one @kernel function: for a sequence of "
            "launches with host code between them, use pygpgpu.Tape")
    return ValueAndGradForm(kernel, **config)


def vjp(kernel:object, **config:object)->VJPForm:
    """The pullback form of a kernel: `result, pullback = pygpgpu.ad.vjp(k)(args)`."""
    from ..kernel.decorator import PyKernel
    if not isinstance(kernel, PyKernel):
        raise TypeError(
            "ad.vjp differentiates one @kernel function: for a sequence of launches "
            "with host code between them, use pygpgpu.Tape")
    return VJPForm(kernel, **config)


def jvp(kernel:object, **config:object)->JVPForm:
    """The forward-mode form of a kernel: `dout = pygpgpu.ad.jvp(k, tangent=dx)(args)`."""
    from ..kernel.decorator import PyKernel
    if not isinstance(kernel, PyKernel):
        raise TypeError(
            "ad.jvp differentiates one @kernel function: for a sequence of launches "
            "with host code between them, use pygpgpu.Tape")
    return JVPForm(kernel, **config)
