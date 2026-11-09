"""Passes that do not need a read-back: bind the array as a deferred device copy.

Between two passes the device already holds the newest bytes. Reading them back and
uploading them again copies twice for nothing unless a host region actually touches the
array. The binding layer already defers a `gnp.ndarray`'s read-back (`_mark_device_written`
plus `to_host`), so an array that only the device touches between two passes is bound
through the same memory with that deferred machinery, and the caller's own array is
brought up to date once, after the last pass.
"""
from __future__ import annotations
import ast
from typing import Dict, List, Optional, Set

import numpy as np


def _touched(names: List[str], statements: List[ast.stmt]) -> Set[str]:
    """The names one host region reads or writes."""
    touched: Set[str] = set()
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                touched.add(node.id)
            elif isinstance(node, ast.Name) and node.id in names:
                touched.add(node.id)
    return touched


def _can_be_deferred(value: object) -> bool:
    """Whether a plain array can travel as its own deferred view safely.

    The deferred path reads the array's bytes back into the view that was bound. A
    non-contiguous array's device layout is a compacted copy (the binding layer stages it),
    so writing those bytes back through the view lands in the wrong place: the caller's
    array looks unmodified and the buffer behind it is scrambled. Such an array keeps the
    immediate read-back it always had.
    """
    return bool(getattr(value.flags, "c_contiguous", False))


def deferred_arrays(plan: object) -> Dict[str, object]:
    """The array arguments a multi-pass body can keep on the device between passes.

    Every argument the passes write is a candidate; one is dropped when a host region or
    the body's tail names it, because those run python on the host and need current bytes.
    """
    regions: List[List[ast.stmt]] = list(plan.launch_meta.get("phase_gaps") or [])
    tail: List[ast.stmt] = list(plan.launch_meta.get("body_tail")
                                or plan.launch_meta.get("phase_tail") or [])
    needed: Set[str] = _touched([], tail)
    for region in regions:
        needed |= _touched([], region)
    deferred: Dict[str, object] = {}
    for name, value in zip(plan.prefix_names, plan.full):
        if name in needed:
            continue
        if isinstance(value, np.ndarray) and value.ndim > 0 and _can_be_deferred(value):
            deferred[name] = value
    return deferred


def deferred_view(value: object) -> Optional[object]:
    """The deferred-read-back view of one array argument, or None."""
    import numpy

    from ...numpy._devices import target_of  # noqa: F401  (kept for symmetry with _to_device)
    from ...numpy.ndarray import ndarray as gnp_ndarray

    if not isinstance(value, numpy.ndarray):
        return None
    return value.view(gnp_ndarray)


def flush_deferred(deferred: Dict[str, object]) -> None:
    """Bring every deferred array's bytes back to the caller's memory, once."""
    from ...numpy.ndarray import ndarray as gnp_ndarray

    for view in deferred.values():
        if isinstance(view, gnp_ndarray):
            view.to_host(wait=True)
