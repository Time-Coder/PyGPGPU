"""The copy-back and the completion constants, split from `kernel/__init__.py`.

`_finish_buffer` is the one place a written buffer's bytes reach the caller —
the ndarray path, the ctypes-record path, and the claim release — and the two
constants name the command-buffer status a poller waits for and how long it
sleeps between reads.
"""
from __future__ import annotations

import ctypes

import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..buffer import Buffer


def _finish_buffer(buffer:"Buffer", host_value:object,
                   need_read_back:object, error:Optional[BaseException]=None,
                   claim_serial:Optional[int]=None)->None:
    """Copy a written buffer back into the caller's array and release its claim.

    `need_read_back` comes from the argument's `ArgInfo` -- the shared read/write
    analysis every backend consults -- rather than from a mask tested here, so a
    buffer the body never writes is not copied back and a written one always is.
    `claim_serial` is the serial captured when the launch claimed the buffer: the
    release validates it, so a poller firing after the caller re-claimed cannot
    free the new holder's claim. The synchronous launch passes both, the
    asynchronous poller the same pair.

    StorageModeShared makes the device's bytes the host's bytes, so the read-back is
    a copy from the buffer's own contents.
    """
    try:
        if need_read_back and error is None and host_value is not None:
            if isinstance(host_value, np.ndarray):
                source = ctypes.string_at(buffer._contents(), buffer.size)
                host_value.reshape(-1).view(np.uint8)[:buffer.size] = np.frombuffer(
                    source, dtype=np.uint8)[:host_value.nbytes]
            elif isinstance(host_value, (ctypes.Structure, ctypes.Union)):
                # A structure receiver (a `self`/`cls` record the launch travels as
                # one buffer): the member writes the body made land in the device
                # bytes, so the read-back writes them into the caller's record --
                # the same "comes home" contract an ndarray argument follows, and
                # the same memmove-into-the-record the OpenCL read-back does.
                source = ctypes.string_at(buffer._contents(), buffer.size)
                target = memoryview(host_value).cast("B")
                target[:min(buffer.size, len(target))] = source[:len(target)]
    finally:
        # The claim is released on every path; a release carrying a serial the buffer
        # no longer holds is a no-op (see MemObject.unuse), so it never shadows the
        # read-back's own error. A raise from the release itself (a buffer that never
        # had a claim) is absorbed: the read-back's failure is the one worth seeing.
        try:
            buffer.unuse(claim_serial)
        except RuntimeError:
            pass


# `MTLCommandBufferStatus`: the value `status` answers once every command in the
# buffer has finished. The poller waits for the buffer to reach it. **The value
# is 4, not 3**: the enumeration is NotEnqueued(0), Enqueued(1), Committed(2),
# Scheduled(3), Completed(4) -- a poller that stops at 3 reads the read-back
# while the GPU is still writing, and every result comes home stale.
_MTCOMMAND_BUFFER_STATUS_COMPLETED:int = 4
# How long the completion poller sleeps between status reads. A launch takes
# microseconds to tens of milliseconds; this is short enough to feel immediate and
# long enough that a sleeping poller costs nothing.
_POLL_INTERVAL:float = 0.0002
