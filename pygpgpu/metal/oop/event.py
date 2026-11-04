"""Events, for the oop layer's launch plumbing.

Metal reports completion through a command buffer's status, not through a
separate event object. The oop layer waits for its command buffer before it
returns (synchronously today), so by the time an Event reaches a caller the work
is already finished; the type therefore records a completed state and offers the
same surface the other backends' events do, so shared code paths work unchanged.

An asynchronous launch (a completion handler) is where a real Event would earn
its keep; when that lands, this class grows a callback list rather than changing
its public shape.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional


class Event:
    """An already-completed event: waiting is a no-op and callbacks run at once."""

    def __init__(self, error:Optional[BaseException]=None)->None:
        self._error:Optional[BaseException] = error
        self._callbacks:List[Callable[["Event", Any], None]] = []

    @property
    def finished(self)->bool:
        return True

    @property
    def status(self)->Any:
        return self._error

    def wait(self)->None:
        """Block until the work this event describes has finished (already true)."""

    def add_callback(self, callback:Callable[["Event", Any], None])->None:
        """Run `callback` now: this event is already complete.

        Metal reports completion through the command buffer's status and the launch
        waits on it before returning, so by the time an `Event` exists there is
        nothing left to wait for. Calling it here rather than storing it is what
        makes the surface behave like the CUDA/HIP streams' `add_callback`.
        """
        callback(self, self._error)

    def on_completed_callbacks_add(self, callback:Callable[["Event", Any], None])->None:
        """Register a completion callback and run it, exactly as `add_callback` does.

        This used to append to a list nothing ever drained: the callback was accepted
        and silently never called, which reads as "registered" to a caller. There is
        no deferred state to wait on (see above), so the two spellings mean the same
        thing here and both run the callback immediately. When a real asynchronous
        launch lands, this is the method that starts deferring -- the list is kept for
        that, and the docstring says so rather than implying it today.
        """
        self._callbacks.append(callback)
        callback(self, self._error)

    @property
    def on_completed_callbacks(self)->List[Callable[["Event", Any], None]]:
        return self._callbacks
