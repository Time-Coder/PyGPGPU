"""The handle a gnp copy returns: a waitable, awaitable completion token.

``gnp.ndarray.to_device()`` and ``gnp.ndarray.to_host()`` are **asynchronous**: they
start the copy and hand back a :class:`Transfer`, which completes once the device has
the bytes (``to_device``) or the host array does (``to_host``). Waiting is the caller's
decision, and the places that genuinely depend on the result wait on their own:

* the engine waits before a copy has to read host bytes that another in-flight copy is
  still writing (see ``ndarray._ensure_host_bytes``);
* a backend waits on the events it recorded for an argument before it thinks about
  reading that argument (the launch path already did this for its read-backs, and the
  same rule now covers the uploads);
* anything that reads the host array -- numpy's own protocol methods, ``arr[i]``,
  ``tolist()`` -- goes through ``_ensure_host_bytes`` first, so a host access after an
  async ``to_device`` is correct without the caller calling ``wait()`` at all.

A Transfer is therefore both a *barrier* (``wait()``) and a *future* (``result()``,
``future()``, ``await transfer``) for the code that wants to overlap.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Callable, List, Optional


class Transfer:
    """One in-flight copy, resolved by the device event that completes it.

    The engine builds one of these per copy and resolves it from the backend's
    completion callback, so the object is safe to hand to another thread unchanged:
    every field is guarded, and callbacks that arrive after completion run at once.
    """

    def __init__(self, description: str = "") -> None:
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[Optional[BaseException]], None]] = []
        self._done = threading.Event()
        self._error: Optional[BaseException] = None
        self.description = description

    # ---------- engine side ----------

    def _complete(self, error: Optional[BaseException] = None) -> None:
        """Mark the copy finished (idempotent) and run everything waiting on it."""
        with self._lock:
            if self._done.is_set():
                return
            self._error = error
            callbacks = list(self._callbacks)
            self._callbacks.clear()
            self._done.set()
        for callback in callbacks:
            try:
                callback(error)
            except BaseException:
                # A waiter's own failure must not leave the remaining waiters hanging.
                pass

    def _on_complete(self, callback: Callable[[Optional[BaseException]], None]) -> None:
        """Run ``callback`` when this copy finishes (immediately if it already has)."""
        with self._lock:
            if not self._done.is_set():
                self._callbacks.append(callback)
                return
            error = self._error
        callback(error)

    # ---------- public side ----------

    @classmethod
    def completed(cls, description: str = "") -> "Transfer":
        """A Transfer that is already finished, for a copy that needed no work."""
        transfer = cls(description)
        transfer._done.set()
        return transfer

    @property
    def done(self) -> bool:
        """Whether the copy has finished (successfully or not)."""
        return self._done.is_set()

    def wait(self, timeout: Optional[float] = None) -> "Transfer":
        """Block until the copy has finished; returns itself so calls can chain."""
        if not self._done.wait(timeout):
            raise TimeoutError(
                f"the copy {self.description or 'requested'} did not finish within "
                f"{timeout} s")
        if self._error is not None:
            raise self._error
        return self

    def result(self, timeout: Optional[float] = None) -> None:
        """Block until finished and raise the copy's error, if it had one."""
        self.wait(timeout)
        return None

    def future(self, kind: str = "concurrent") -> Any:
        """A standard future that mirrors this transfer.

        ``kind="concurrent"`` returns a ``concurrent.futures.Future``;
        ``kind="asyncio"`` returns an ``asyncio.Future`` bound to the running loop.
        The transfer itself still owns the copy; the future only mirrors it.
        """
        if kind == "asyncio":
            loop = asyncio.get_running_loop()
            future: Any = loop.create_future()

            def settle(error: Optional[BaseException]) -> None:
                if error is None:
                    loop.call_soon_threadsafe(future.set_result, None)
                else:
                    loop.call_soon_threadsafe(future.set_exception, error)

            self._on_complete(settle)
            return future

        if kind != "concurrent":
            raise ValueError(
                f"future() takes kind='concurrent' or kind='asyncio', got {kind!r}")
        future = concurrent.futures.Future()

        def settle_concurrent(error: Optional[BaseException]) -> None:
            if error is None:
                future.set_result(None)
            else:
                future.set_exception(error)

        self._on_complete(settle_concurrent)
        return future

    def __await__(self):
        return self.future("asyncio").__await__()

    def __repr__(self) -> str:
        state = "done" if self.done else "pending"
        description = f" {self.description}" if self.description else ""
        return f"Transfer({state}{description})"
