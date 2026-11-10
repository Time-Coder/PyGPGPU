# The process-wide int=/float= width mapping and its lock.
from __future__ import annotations
import threading
from typing import (
    Dict,
    Optional,
)
from ..translator import (
    validate_type_map,
)



# The process-wide `int`/`float` mapping `config.type_map` installs, or None. Read **once**,
# when a kernel is decorated: a mapping is part of what that kernel means, so changing it
# afterwards cannot change a kernel that already exists (its `type_map` is a snapshot, and it
# is in its cache key). That is also what keeps `config.type_map` honest across a warm cache.
_type_map_default:Optional[Dict[str, str]] = None
_type_map_lock = threading.Lock()


def _install_type_map(declared:Optional[Dict[str, object]])->None:
    """Set the default device width a python ``int`` / ``float`` travels as.

    ``declared`` is keyed by the internal name (``"int"`` / ``"float"``); ``None`` restores
    the built-in widths (``int`` = 32-bit ``int``, ``float`` = 32-bit ``float``).

    The values are the same type objects `@kernel(float=...)` takes -- a numpy dtype or
    scalar type, a ctypes scalar class, or a device type name. A decorator's own `int=`/
    `float=` wins over this, and a numpy scalar keeps its own width either way.

    **It applies at decoration time, not at call time.** A kernel that already exists keeps
    the mapping it was declared with, including across a warm cache and across processes:
    the mapping is in the generated source's digest and in the C name, so a later setting
    cannot be served the binary an earlier one compiled. Call it before decorating.
    """
    global _type_map_default
    if not declared:
        with _type_map_lock:
            _type_map_default = None
        return
    resolved:Dict[str, str] = validate_type_map(declared)
    with _type_map_lock:
        _type_map_default = resolved


def _type_map_default_snapshot()->Dict[str, str]:
    """The process-wide mapping `config.type_map` installed, as device type names."""
    with _type_map_lock:
        current:Optional[Dict[str, str]] = _type_map_default
    return dict(current) if current else {}


def _effective_type_map(type_map:Optional[Dict[str, object]])->Dict[str, str]:
    """The mapping a kernel is declared with: its own, over the process-wide default."""
    base:Dict[str, str] = _type_map_default_snapshot()
    if not type_map:
        return base
    resolved:Dict[str, str] = validate_type_map(type_map)
    base.update(resolved)
    return base
