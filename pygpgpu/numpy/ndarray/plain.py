# The ufunc dispatch and the plain-python reading of an array's values.
from __future__ import annotations

from typing import Dict, Mapping, Tuple
import numpy as np



def _to_plain(values: object) -> object:
    """Replace a gnp array by its plain numpy view, recursing into a tuple or list."""
    # The class lives in the package facade, which imports this module: the import is made
    # where it is used, so the two do not import each other at module level.
    from . import ndarray
    if isinstance(values, ndarray):
        values._ensure_host_bytes()
        return values.view(np.ndarray)
    if isinstance(values, tuple):
        return tuple(_to_plain(item) for item in values)
    if isinstance(values, list):
        return [_to_plain(item) for item in values]
    return values


def _to_plain_args(values: Tuple[object, ...]) -> Tuple[object, ...]:
    """Every positional argument's plain view, as a tuple a call can be splatted with."""
    return tuple(_to_plain(item) for item in values)


def _to_plain_mapping(mapping: Mapping[str, object]) -> Dict[str, object]:
    return {key: _to_plain(value) for key, value in mapping.items()}


def _aliases(value: object, original: object) -> bool:
    """Whether ``value`` overlaps ``original``'s memory.

    Both are gnp elements, so the question is asked of the plain buffers they index:
    asking numpy directly would dispatch ``np.shares_memory`` back into
    ``__array_function__``, which is the very method that may be asking.
    """
    return np.shares_memory(value.view(np.ndarray), original.view(np.ndarray),
                            max_work=2 ** 20)


def _call_ufunc(ufunc: np.ufunc, method: str, inputs: Tuple[object, ...],
                kwargs: Mapping[str, object]) -> object:
    """Run a ufunc with plain arrays, keeping the method (``__call__``, ``reduce``, ...).

    numpy refuses to dispatch a ufunc whose inputs are its own subclass back to the same
    subclass a second time, so the operation is evaluated on the plain views instead of
    through ``super().__array_ufunc__``.
    """
    if method == "__call__":
        return ufunc(*inputs, **kwargs)
    return getattr(ufunc, method)(*inputs, **kwargs)
