# The per-launch frame, the thread pool the async forms use, and their futures.
from __future__ import annotations
import asyncio
import concurrent.futures
import threading
from typing import Optional, Tuple



class PythonFrame:
    """The launch state the builtin implementations read while a body runs serially.

    Only launch state lives here: the launch shape and the current work item. The operations
    themselves are the bodies in ``builtins.py``. One frame is installed per thread for the
    duration of a kernel call, so two concurrent launches never see each other's work item.
    """

    def __init__(self, global_size: Tuple[int, ...] = (),
                 local_size: Tuple[int, ...] = ()) -> None:
        self.global_size: Tuple[int, ...] = tuple(int(g) for g in global_size)
        self.local_size: Tuple[int, ...] = tuple(int(l) for l in local_size)
        # Global id per dimension of the work item being run; None for the plain call of
        # a magic-loop body, where there is no current work item.
        self.coords: Optional[Tuple[int, ...]] = None



_POOL_LOCK = threading.Lock()


_POOL_LOCK = threading.Lock()
_POOL: Optional[concurrent.futures.ThreadPoolExecutor] = None



def _pool() -> concurrent.futures.ThreadPoolExecutor:
    """The shared executor behind submit()/async_call() (created on first use)."""
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = concurrent.futures.ThreadPoolExecutor(
                    thread_name_prefix="pygpgpu-python")
    return _POOL



def shutdown_thread_pool(wait: bool = True) -> None:
    """Shut the shared executor down (it is recreated on the next submit/async_call)."""
    global _POOL
    with _POOL_LOCK:
        pool: Optional[concurrent.futures.ThreadPoolExecutor] = _POOL
        _POOL = None
    if pool is not None:
        pool.shutdown(wait=wait)



def _current_loop() -> asyncio.AbstractEventLoop:
    """The running loop, else the thread's current loop, else a fresh installed one.

    ``async_call`` cannot be awaited outside a loop anyway, but returning a future
    bound to an installed loop still lets a caller drive it with
    ``loop.run_until_complete(future)`` instead of failing at creation time.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass
    loop: Optional[asyncio.AbstractEventLoop] = None
    try:
        loop = asyncio.get_event_loop()
    except Exception:
        loop = None  # no loop is set on this thread (or reading it is no longer allowed)
    if loop is not None and not loop.is_closed():
        return loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop



def _asyncio_future(work: concurrent.futures.Future) -> asyncio.Future:
    """Bridge a pool future to an asyncio.Future, resolving it on the event loop thread.

    The value the work produced is carried over, so a launch that returns something
    resolves its asyncio future with that value, exactly as the pool future does.
    """
    loop: asyncio.AbstractEventLoop = _current_loop()
    future: asyncio.Future = loop.create_future()

    def complete(done: concurrent.futures.Future) -> None:
        error: Optional[BaseException] = None
        result: object = None
        try:
            error = done.exception()
            if error is None:
                result = done.result()
        except BaseException as exc:  # cancelled or failed to deliver a result
            error = exc

        def settle() -> None:
            if future.done():
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)

        try:
            loop.call_soon_threadsafe(settle)
        except RuntimeError:
            pass  # the loop was closed before the launch finished

    work.add_done_callback(complete)
    return future
