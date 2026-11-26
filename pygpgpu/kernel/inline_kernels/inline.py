# The `inline` singleton: the outermost `with` item, and the catcher of a block.
"""`with inline, kernel(...):` needs two objects, and this is the outer one.

The inner item (`kernel(...)`, a `_KernelSpec`) is entered second and raises
:class:`~.site.InlineBlock` from its `__enter__`; this singleton's `__exit__` catches that
exception, which is what suppresses the body. Suppressing it is the mechanism, not a trick: a
`with` body only runs when its head returns normally, so raising from the last head item and
returning True here means the caller's frame never executes a line of device code.

`inline` must therefore come **first**. `with kernel(...), inline:` enters the manager first,
so its exception escapes before this catcher is on the stack -- the failure its own `__exit__`
names. `inline` alone is refused too, because the block would otherwise silently do nothing.
"""
from __future__ import annotations

import threading
from types import TracebackType
from typing import List, Optional, Type

from .site import InlineBlock


class _InlineResult:
    """What `with inline as result, kernel(...):` binds.

    A kernel body may claim a result (`return value` inside its magic loop, or a host
    `return` after it), exactly as a `@kernel` body may. A `with` statement has no value of
    its own, so the answer is delivered on the object this binds: `result.value` is the
    kernel's return value once the statement has finished.
    """

    def __init__(self)->None:
        self.value:object = None

    def __repr__(self)->str:
        return f"<inline kernel result: {self.value!r}>"


class _Inline:
    """The fixed singleton exported as `inline`.

    Fixed, not a factory: it is entered before the block manager exists, and its only job is
    to be the outermost catcher of the block signal.

    `__enter__` returns the result slot the block fills. It cannot live on `self` and be read
    back in `__exit__` -- two threads launching at once would share it -- so it is kept on a
    per-thread stack, pushed by `__enter__` and popped by `__exit__`. The stack is per thread
    and `with` statements nest, so one thread's slots are never another's.
    """

    def __enter__(self)->_InlineResult:
        slot:_InlineResult = _InlineResult()
        _slots().append(slot)
        return slot

    def __exit__(self, exc_type:Optional[Type[BaseException]],
                 exc:Optional[BaseException],
                 traceback:Optional[TracebackType])->bool:
        stack:List[_InlineResult] = _slots()
        slot:Optional[_InlineResult] = stack.pop() if stack else None
        if exc_type is InlineBlock:
            site:object = exc
            assert site is not None
            if site.consumed:                       # a re-raise must not launch twice
                return True
            site.consumed = True
            try:
                value:object = _launch_site(site)
            finally:
                # The frame is only needed to read this call's captured values; holding it
                # afterwards would keep a finished caller alive for the rest of the process.
                site.frame = None
            if slot is not None:
                slot.value = value
            return True
        if exc_type is None:
            raise SyntaxError(
                "`inline` must be paired with `kernel(...)`: the block is a kernel body, and "
                "without a `kernel(...)` head there is nothing to launch. Write "
                "`with inline, kernel(...):` and put the body under it")
        return False                                # a real error from the head propagates


def _launch_site(site:InlineBlock)->object:
    """Launch one captured block, through the cache that gives one kernel per site."""
    from .launch import launch_block

    return launch_block(site)


_SLOTS_BY_THREAD = threading.local()


def _slots()->List[_InlineResult]:
    """This thread's stack of result slots, created on first use."""
    stack:Optional[List[_InlineResult]] = getattr(_SLOTS_BY_THREAD, "stack", None)
    if stack is None:
        stack = []
        _SLOTS_BY_THREAD.stack = stack
    return stack


# The singleton. It is exported from `pygpgpu` and `pygpgpu.kernel`, and it must be the one
# object every block compares against -- `_find_with` uses identity, not a name or a class.
INLINE:_Inline = _Inline()
inline:_Inline = INLINE
