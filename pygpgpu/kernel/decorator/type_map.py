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



# The process-wide `int`/`float` mapping `set_type_map()` installs, or None. Read **once**,
# when a kernel is decorated: a mapping is part of what that kernel means, so changing it
# afterwards cannot change a kernel that already exists (its `type_map` is a snapshot, and it
# is in its cache key). That is also what keeps `set_type_map()` honest across a warm cache.
_type_map_default:Optional[Dict[str, str]] = None
_type_map_lock = threading.Lock()


def set_type_map(int:object=None, float:object=None)->None:
    """Set the default device width a python ``int`` / ``float`` travels as.

    Called with no arguments it restores the built-in widths (``int`` = 32-bit ``int``,
    ``float`` = 32-bit ``float``)::

        set_type_map(float=np.float64)     # every kernel decorated from now on is 64-bit

        @kernel
        def accurate(scale, out): ...       # `scale` is a `double`

    The values are the same type objects `@kernel(float=...)` takes -- a numpy dtype or
    scalar type, a ctypes scalar class, or a device type name. A decorator's own `int=`/
    `float=` wins over this, and a numpy scalar keeps its own width either way.

    **It applies at decoration time, not at call time.** A kernel that already exists keeps
    the mapping it was declared with, including across a warm cache and across processes:
    the mapping is in the generated source's digest and in the C name, so a later setting
    cannot be served the binary an earlier one compiled. Call it before decorating.
    """
    global _type_map_default
    if int is None and float is None:
        with _type_map_lock:
            _type_map_default = None
        return
    declared:Dict[str, object] = {}
    if int is not None:
        declared["int"] = int
    if float is not None:
        declared["float"] = float
    resolved:Dict[str, str] = validate_type_map(declared)
    with _type_map_lock:
        _type_map_default = resolved


def get_type_map()->Dict[str, str]:
    """The process-wide mapping `set_type_map()` installed, as device type names."""
    with _type_map_lock:
        current:Optional[Dict[str, str]] = _type_map_default
    return dict(current) if current else {}


def _effective_type_map(type_map:Optional[Dict[str, object]])->Dict[str, str]:
    """The mapping a kernel is declared with: its own, over the process-wide default."""
    base:Dict[str, str] = get_type_map()
    if not type_map:
        return base
    resolved:Dict[str, str] = validate_type_map(type_map)
    base.update(resolved)
    return base
