"""Name <-> object tables for the dual-sided builtins.

The translator recognizes a builtin call by resolving the callee and looking the resulting
*object* up here, never by matching the spelling of the name. That is what keeps a user's
own ``def dot(...)`` an ordinary device helper while ``from pygpgpu import dot`` still
translates as the builtin, however the name was imported.

``builtins.py`` is the user-facing module; this one is its index and is not part of the
public API.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from . import builtins as _namespace

# name -> the function object published under it.
BUILTIN_FUNCTIONS: Dict[str, Any] = {
    name: getattr(_namespace, name) for name in _namespace.__all__}

# The same table the other way round, keyed by the function object itself.
BUILTIN_NAMES_BY_FUNCTION: Dict[Any, str] = {
    function: name for name, function in BUILTIN_FUNCTIONS.items()}


def builtin_name_of(target: Any) -> Optional[str]:
    """The builtin name ``target`` is, or None when it is not one of ours.

    Used for a call whose callee was resolved through the kernel's globals: the answer is
    what decides between "this is the builtin" and "this is a device helper of the user's".
    """
    return BUILTIN_NAMES_BY_FUNCTION.get(target)


def builtin_function(name: str) -> Optional[Any]:
    """The function published as ``name``, or None when there is no such builtin."""
    return BUILTIN_FUNCTIONS.get(name)
