# The dtype and async helpers the kernel modules share.
from __future__ import annotations
import asyncio
import concurrent.futures
import threading
from typing import Optional
import numpy as np



def complex_of_dtype(dtype:object)->object:
    """The complex dtype behind a numpy record layout of two reals, or None.

    A complex array's bytes *are* the record's bytes: `complex64` is a `float` pair and
    `complex128` a `double` pair, in that order. So an argument declared against one of the
    package's fixed complex structures binds a complex array unchanged, and the upload and
    read-back have to speak the *complex* dtype -- a record and a complex array will not
    assign to one another under numpy's casting rules.
    """
    if not isinstance(dtype, np.dtype) or dtype.fields is None:
        return None
    if tuple(dtype.names or ()) != ("re", "im"):
        return None
    first:np.dtype = np.dtype(dtype.fields["re"][0])
    if first != np.dtype(dtype.fields["im"][0]):
        return None
    if first.name == "float32":
        return np.dtype(np.complex64)
    if first.name == "float64":
        return np.dtype(np.complex128)
    return None



_async_executor:Optional[concurrent.futures.ThreadPoolExecutor] = None



_async_executor_lock = threading.Lock()



def _async_workers()->concurrent.futures.ThreadPoolExecutor:
    """Lazily create the worker pool that finishes asynchronous launches.

    A launch that finds every pooled stream busy is completed here instead of on
    the caller's thread, so submit()/async_call() never block the caller.
    """
    global _async_executor
    with _async_executor_lock:
        if _async_executor is None:
            _async_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="pygpgpu-hip")
        return _async_executor



def _set_future_error(future:object, error:BaseException)->None:
    """Deliver an exception to an asyncio or concurrent future from any thread."""
    if isinstance(future, asyncio.Future):
        loop = future.get_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(future.set_exception, error)
            return
    future.set_exception(error)



def _deliver(future:object, pending:concurrent.futures.Future)->Optional[BaseException]:
    """Move a completed stream future onto the caller's future; return any error."""
    try:
        error = pending.exception()
    except BaseException as caught:
        error = caught
    if error is None:
        if isinstance(future, asyncio.Future):
            loop = future.get_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(future.set_result, None)
            else:
                future.set_result(None)
        else:
            future.set_result(None)
    else:
        _set_future_error(future, error)
    return error



def _structure_bytes(value)->bytes:
    """The raw bytes of one record value, from any of the shapes a caller passes.

    A `ctypes` instance converts through `bytes()`, which is what backs it. A numpy record
    does not: `bytes(record)` goes through the buffer protocol, and a dtype whose members
    overlap (a union layout) or whose offsets do not ascend is refused there with "dtypes
    with overlapping or out-of-order fields are not representable as buffers". `tobytes()`
    copies the element's raw memory instead, so it answers for every layout.
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, np.void):
        return value.tobytes()
    return bytes(value)
