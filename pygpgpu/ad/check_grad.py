# `check_grad`: the AD answer against finite differences, on the machine with no GPU.
#
# The `python` backend runs the adjoint body as written, so it *is* a reference
# implementation of the gradient -- which is what makes AD correctness testable in full
# without a device. This function is a first-class deliverable, not a test helper: the
# adjoint gradients run on whatever backend the caller asks for, the finite differences
# run on the interpreter, and a mismatch raises with the offending elements named.
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .forms import GradForm, RETURN_SPELL


def check_grad(kernel:object, *args:object, wrt:Optional[Sequence[str]] = None,
               loss:object = RETURN_SPELL, backend:Optional[str] = None,
               eps:float = 1e-3, rtol:float = 1e-2, atol:float = 1e-3)->float:
    """Compare the AD gradients against central finite differences.

    The adjoint gradients are computed on ``backend`` (the interpreter when None), and
    every tracked element is perturbed by ``eps`` both ways, running the kernel on the
    `python` backend for the loss value. Returns the largest absolute deviation; raises
    :class:`AssertionError` naming the worst element when a gradient is outside
    ``atol + rtol * |fd|``.
    """
    form:GradForm = GradForm(kernel, backend=backend, wrt=wrt, loss=loss)
    arguments:List[Tuple[str, object]] = form._bind(args, {})
    wrt_names:List[str] = form._resolve_wrt(arguments)
    values:List[object] = form._positional(arguments)
    param_names:List[str] = form._params()
    # The gradients accumulate into `.grad`; a check must measure one call, so it clears.
    for wrt_name in wrt_names:
        array:object = values[param_names.index(wrt_name)]
        zero = getattr(array, "zero_grad", None)
        if zero is not None:
            zero()
    gradients:Tuple[object, ...] = form()(*args)
    loss_slot:Optional[object] = values[param_names.index(loss)] \
        if loss != RETURN_SPELL else None
    worst:float = 0.0
    for wrt_name, gradient in zip(wrt_names, gradients):
        array:object = values[param_names.index(wrt_name)]
        fd:object = _finite_differences(kernel, values, param_names, wrt_name, array,
                                        loss, loss_slot, eps)
        ad:object = np.asarray(_to_host(gradient), dtype=np.float64)
        fd64:object = np.asarray(fd, dtype=np.float64)
        if ad.shape != fd64.shape:
            raise AssertionError(
                f"check_grad({kernel.__name__}): the gradient of `{wrt_name}` has shape "
                f"{ad.shape} but the finite differences have {fd64.shape}")
        bad:object = np.abs(ad - fd64) - (atol + rtol * np.abs(fd64))
        index:object = np.unravel_index(np.argmax(bad), bad.shape) if bad.size else None
        worst_here:float = float(bad[index]) if index is not None else 0.0
        worst = max(worst, worst_here)
        if worst_here > 0:
            raise AssertionError(
                f"check_grad({kernel.__name__}): the gradient of `{wrt_name}` disagrees "
                f"with finite differences at {index}: AD={ad[index]!r}, FD={fd64[index]!r} "
                f"(tolerance atol={atol}, rtol={rtol})")
    return worst


def _finite_differences(kernel:object, values:List[object], param_names:List[str],
                        wrt_name:str, array:object, loss:object,
                        loss_slot:Optional[object], eps:float)->object:
    """Central differences of the loss w.r.t. one array's elements, on the interpreter."""
    shape:object = np.asarray(array).shape
    flat:object = np.asarray(_to_host(array)).copy()
    out:object = np.zeros(flat.size, dtype=np.float64)
    for offset in range(flat.size):
        plus:List[object] = list(values)
        minus:List[object] = list(values)
        plus[param_names.index(wrt_name)] = _perturbed(array, flat, offset, +eps)
        minus[param_names.index(wrt_name)] = _perturbed(array, flat, offset, -eps)
        if loss_slot is not None:
            _zero_loss(loss_slot)
        high:object = _run_loss(kernel, plus, loss, loss_slot)
        if loss_slot is not None:
            _zero_loss(loss_slot)
        low:object = _run_loss(kernel, minus, loss, loss_slot)
        out.flat[offset] = (high - low) / (2.0 * eps)
    return out.reshape(shape)


def _perturbed(array:object, flat:object, offset:int, delta:float)->object:
    """The array with one element moved, as a plain numpy array the call can pass."""
    perturbed:object = np.array(flat, dtype=np.asarray(array).dtype, copy=True)
    perturbed.flat[offset] = float(flat.flat[offset]) + delta
    return perturbed


def _run_loss(kernel:object, values:List[object], loss:object,
              loss_slot:Optional[object])->float:
    """One forward run on the interpreter, answering the loss value."""
    result:object = kernel["python"](*values)
    if loss == RETURN_SPELL:
        return float(result)
    return float(np.asarray(_to_host(loss_slot)).reshape(-1)[0])


def _zero_loss(loss_slot:object)->None:
    """Clear the loss array before a run that accumulates into it."""
    plain:object = _to_host(loss_slot)
    view:object = plain if isinstance(plain, np.ndarray) else np.asarray(plain)
    view.fill(0)


def _to_host(value:object)->object:
    """The plain numpy view of a gnp array (its bytes brought home), or the value."""
    ensure = getattr(value, "_ensure_host_bytes", None)
    if ensure is not None:
        ensure()
    view = getattr(value, "view", None)
    if view is not None and isinstance(value, np.ndarray):
        return value.view(np.ndarray)
    return value
