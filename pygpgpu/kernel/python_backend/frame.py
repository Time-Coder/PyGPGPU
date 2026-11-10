# The per-launch frame, the thread pool the async forms use, and their futures.
from __future__ import annotations
import asyncio
import concurrent.futures
import threading
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    # The element type is numpy's own; naming it here keeps this module import-light while
    # still annotating the storage a block-local buffer keeps.
    import numpy as np
    from numpy.typing import DTypeLike



class PythonFrame:
    """The launch state the builtin implementations read while a body runs serially.

    Only launch state lives here: the launch shape, the current work item, and the
    work-group storage the body has allocated so far. The operations themselves are the
    bodies in ``builtins.py``. One frame is installed per thread for the duration of a
    kernel call, so two concurrent launches never see each other's work item.
    """

    def __init__(self, global_size: Tuple[int, ...] = (),
                 local_size: Tuple[int, ...] = ()) -> None:
        self.global_size: Tuple[int, ...] = tuple(int(g) for g in global_size)
        self.local_size: Tuple[int, ...] = tuple(int(l) for l in local_size)
        # Global id per dimension of the work item being run; None for the plain call of
        # a magic-loop body, where there is no current work item.
        self.coords: Optional[Tuple[int, ...]] = None
        # True only for a launch that runs the body once *per work item* -- the gid style.
        # A magic-loop body runs once and stands for the whole launch, so the block-local
        # builtins (whose value is the caller's own block) have no meaning there.
        self.per_work_item: bool = False
        # The work-group arrays this launch has handed out, keyed by (call site, block id).
        # A serial run still has to give every work item of one block the *same* array:
        # that is what a work group's shared memory is, and it is what lets a body read a
        # slot a different work item wrote.
        self.shared: Dict[Tuple[str, Tuple[int, ...]], "np.ndarray"] = {}


    def block_of(self, coords: Tuple[int, ...]) -> Tuple[int, ...]:
        """The work-group id a global position belongs to, one index per launch axis."""
        block: list = []
        for axis, coordinate in enumerate(coords):
            local: int = self.local_size[axis] if axis < len(self.local_size) else 1
            block.append(int(coordinate) // max(1, local))
        return tuple(block)


    def shared_array(self, site: str, block: Tuple[int, ...], shape: Tuple[int, ...],
                     dtype: "DTypeLike") -> "np.ndarray":
        """The array this block owns at one creation site, created on first use.

        The site is part of the key because one body may declare several buffers; the block
        id is the other half, because that is which group's storage is meant.
        """
        import numpy as np

        key: Tuple[str, Tuple[int, ...]] = (site, tuple(int(b) for b in block))
        store: "np.ndarray" = self.shared.get(key)
        if store is None:
            store = np.zeros(tuple(int(s) for s in shape), dtype=dtype)
            self.shared[key] = store
        return store



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
