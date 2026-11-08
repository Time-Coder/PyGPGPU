# The dtype and async helpers the kernel modules share.
from __future__ import annotations
import asyncio
import concurrent.futures
import threading
from typing import Optional, Tuple
import numpy as np
from ctypes import sizeof


def complex_of_dtype(dtype: object) -> object:
    """The complex dtype behind a numpy record layout of two reals, or None.

    A complex array's bytes *are* the record's bytes: `complex64` is a `float` pair and
    `complex128` a `double` pair, in that order. So an argument declared against one of the
    package's fixed complex structures can bind a complex array unchanged, and everything the
    copy path does with it -- the upload, the read-back -- has to speak the *complex* dtype
    rather than the record's, or the element-wise copies refuse to assign.
    """
    if not isinstance(dtype, np.dtype) or dtype.fields is None:
        return None
    names:Tuple[str, ...] = tuple(dtype.names or ())
    if names != ("re", "im"):
        return None
    first:np.dtype = np.dtype(dtype.fields["re"][0])
    second:np.dtype = np.dtype(dtype.fields["im"][0])
    if first != second:
        return None
    if first.name == "float32":
        return np.dtype(np.complex64)
    if first.name == "float64":
        return np.dtype(np.complex128)
    return None



def _host_dtype_for(content_type: object) -> object:
    """The dtype a host array must have to bind this device element type.

    A complex structure binds a *complex* array, not an array of records: the two layouts are
    identical, and numpy will not assign a record into a complex array (nor the reverse) under
    the casting rule a read-back uses. Answering the complex dtype here is what keeps the
    upload and the read-back working for a `c64*` parameter.
    """
    if isinstance(content_type, type) and hasattr(content_type, "dtype"):
        return complex_of_dtype(content_type.dtype) or content_type.dtype
    return content_type



def _element_dtype(content_type: object) -> object:
    """The numpy dtype an array argument's elements should be cast to.

    `char` binds as `ctypes.c_char`, and numpy reads that as a one-byte *string*: casting an
    integer array to it converts the numbers to characters (a 7 becomes the byte for "7",
    0x37), which the device then reads as 55. `np.dtype(c_char)` is `|S1` however it is
    spelled, so the string kinds are mapped back to the integer of the same width.
    """
    if isinstance(content_type, np.dtype):
        dtype: np.dtype = content_type
    else:
        size = sizeof(content_type) if hasattr(content_type, "__mro__") else 0
        if size == 1:
            # c_char / c_ubyte: a signedness guess would be worse than the kernel's own
            # declaration, and both are one byte wide, so an unsigned byte is the safe
            # carrier and the kernel reads it back as its declared type.
            dtype = np.dtype(np.uint8)
        else:
            dtype = np.dtype(content_type)
    if dtype.kind in ("S", "U", "O"):
        return np.dtype(f"i{dtype.itemsize}")
    return dtype



_async_executor:Optional[concurrent.futures.ThreadPoolExecutor] = None



_async_executor_lock = threading.Lock()



def _async_workers()->concurrent.futures.ThreadPoolExecutor:
    """Lazily create the worker pool that completes asynchronous launches.

    A launch that finds every pooled queue busy is finished here rather than on
    the caller's thread, so submit()/async_call() never block.
    """
    global _async_executor
    with _async_executor_lock:
        if _async_executor is None:
            _async_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="pygpgpu-opencl")
        return _async_executor



def _set_error(future:object, error:BaseException)->None:
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
        _set_error(future, error)
    return error



def _bridge(completion:object, future:object, on_done:object)->None:
    """Mirror a completion future onto the caller's and then run ``on_done``.

    ``on_done`` returns the queue to the pool, so it must wait until every event
    the launch registered has completed.
    """
    def relay(pending:concurrent.futures.Future)->None:
        _deliver(future, pending)
        on_done()

    completion.add_done_callback(relay)



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
