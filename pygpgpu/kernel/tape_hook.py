# The slot an active `Tape` records launches through.
#
# A tape records kernel launches only, and the launch is the unit the decorator already
# owns -- so the recording point is one call at the end of a successful launch. This
# module exists so that call is cheap when no tape is active: it is a two-line module the
# decorator imports at load, holding one module-level slot and one check, and it never
# imports `pygpgpu.ad` (which would put the AST transform on the plain-launch import
# path). `Tape.__enter__` fills the slot; the check reads it.
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# The Tape currently recording, set on enter and cleared on exit. One slot, not a
# thread-local: a `with Tape():` block owns the launches of the code inside it, the same
# way a global flag owns a region, and a nested tape is refused rather than silently
# shadowed.
active:Optional[Any] = None


def record(kernel:object, args:Tuple[object, ...], kwargs:Dict[str, object]) -> None:
    """Record one launch on the active tape, if there is one."""
    tape:Optional[Any] = active
    if tape is not None:
        tape.record(kernel, args, kwargs)
