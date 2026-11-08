from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np

from ..host_runtime import atomic_in_place

def atomic_add(array: "np.ndarray[Any, Any]",
            index: Union[int, Tuple[int, ...]],
            value: "np.number[Any]") -> object:
    """Add ``value`` to ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: current + value)


def atomic_sub(array: "np.ndarray[Any, Any]",
            index: Union[int, Tuple[int, ...]],
            value: "np.number[Any]") -> object:
    """Subtract ``value`` from ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: current - value)


def atomic_min(array: "np.ndarray[Any, Any]",
            index: Union[int, Tuple[int, ...]],
            value: "np.number[Any]") -> object:
    """Keep the smaller of ``array[index]`` and ``value``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: min(current, value))


def atomic_max(array: "np.ndarray[Any, Any]",
            index: Union[int, Tuple[int, ...]],
            value: "np.number[Any]") -> object:
    """Keep the larger of ``array[index]`` and ``value``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: max(current, value))


def atomic_and(array: "np.ndarray[Any, Any]",
            index: Union[int, Tuple[int, ...]],
            value: "np.number[Any]") -> object:
    """Bitwise-and ``value`` into ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: current & value)


def atomic_or(array: "np.ndarray[Any, Any]",
            index: Union[int, Tuple[int, ...]],
            value: "np.number[Any]") -> object:
    """Bitwise-or ``value`` into ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: current | value)


def atomic_xor(array: "np.ndarray[Any, Any]",
            index: Union[int, Tuple[int, ...]],
            value: "np.number[Any]") -> object:
    """Bitwise-xor ``value`` into ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: current ^ value)


def atomic_xchg(array: "np.ndarray[Any, Any]",
            index: Union[int, Tuple[int, ...]],
            value: "np.number[Any]") -> object:
    """Store ``value`` into ``array[index]``; returns the previous value."""
    return atomic_in_place(array, index, lambda current: value)


def atomic_cas(array: object, index: int, compare: object, value: object) -> object:
    """Store ``value`` into ``array[index]`` when it still holds ``compare``.

    Returns the previous value, so a caller can tell whether the swap happened.
    """
    return atomic_in_place(
        array, index, lambda current: value if current == compare else current)
