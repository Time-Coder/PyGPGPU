# The future plumbing the async launch forms share.
from __future__ import annotations
import asyncio
from typing import Optional



def settle_future(future:object, error:Optional[BaseException]=None) -> None:
    """Complete an asyncio or concurrent future from any thread.

    Driver completion callbacks run on a driver thread, not on the event loop's.
    Setting the result directly does not wake a loop that is blocked in
    ``select()``, so a plain ``await`` on such a future hangs even though the work
    finished -- only ``asyncio.wait_for`` appeared to work, because it wraps the
    future in a Task whose completion re-schedules the loop. Scheduling the
    completion through the loop keeps a bare await responsive.
    """
    if isinstance(future, asyncio.Future):
        loop = future.get_loop()
        if loop.is_running():
            if error is None:
                loop.call_soon_threadsafe(future.set_result, None)
            else:
                loop.call_soon_threadsafe(future.set_exception, error)
            return
    if error is None:
        future.set_result(None)
    else:
        future.set_exception(error)


def _asyncio_future_of(asyncio_module:object) -> object:
    """An ``asyncio.Future`` bound to a loop that exists, even off the loop's thread.

    ``asyncio.Future()`` demands a running (or current) event loop, and a ``submit``-
    style call may come from a thread that has neither -- the future is then meant for
    a loop the caller awaits on later. The python backend solves this with
    ``_current_loop`` (install a fresh loop when none is set); this is the same rule,
    callable without importing asyncio in the caller. Kept private: it exists so a
    backend's ``async_call`` can spell ``future_type()`` for both future kinds.
    """
    current_loop:object = _current_loop()
    return asyncio_module.Future(loop=current_loop)


def _current_loop() -> object:
    """The running loop, else the thread's current loop, else a fresh installed one.

    The same contract the python backend's own bridge states: ``async_call`` cannot
    be awaited outside a loop anyway, but returning a future bound to an installed
    loop still lets a caller drive it with ``loop.run_until_complete(future)``
    instead of failing at creation time.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass
    loop:Optional[asyncio.AbstractEventLoop] = None
    try:
        loop = asyncio.get_event_loop()
    except Exception:
        loop = None  # no loop is set on this thread (or reading it is no longer allowed)
    if loop is not None and not loop.is_closed():
        return loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop
