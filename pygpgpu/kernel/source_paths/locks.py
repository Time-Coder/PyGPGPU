# The locks around a generated file, the timings they use, and the directory they need.
from __future__ import annotations
import contextlib
import io
import os
import threading
import time
from typing import Dict, Iterator



# One generated file belongs to one python module, and several threads (and several
# processes) can compile different kernels of that module at the same time. The write is a
# read-modify-write -- read the file, merge the definitions this call rendered into it, and
# write it back -- so without a lock the later writer drops what the earlier one added, and
# the kernel that lost its definition fails to load with a bare `KeyError` naming the entry
# point it cannot find. Two locks are needed and neither replaces the other:
#
# * a process-wide lock per file (the table below), because a `threading.Lock` is what makes
#   this atomic between threads, and threads in one process share every file;
# * an OS advisory lock on a `.lock` file beside the target (see `_file_lock`), because
#   separate processes share nothing in memory -- the `__clcache__` directory is on disk and
#   every process writing the same module writes the same path. The lock is the OS's, not a
#   convention of ours, so a process that dies releases it (see `_try_lock`).
#
# The process lock is re-entrant: a kernel that reaches a helper compiles both, and the
# helper's own write may land inside the caller's critical section.
_WRITE_LOCKS: Dict[str, threading.RLock] = {}


_WRITE_LOCKS_GUARD = threading.Lock()


# The files whose lock *this thread* already holds, so `_file_lock` can re-enter rather
# than wait on itself. Thread-local by construction: a second thread must wait, and only
# the owner re-enters.
_file_lock_state = threading.local()


# How long a write waits for another process, and how long one attempt of the lock file
# lasts. A compile is seconds at worst; ten minutes is "hang detection", not a budget.
_FILE_LOCK_TIMEOUT: float = 600.0


_FILE_LOCK_POLL: float = 0.05


_MKDIR_ATTEMPTS: int = 8
# How long one generated-source write keeps retrying while another process holds the
# destination. A scanner's hold is milliseconds and an indexer's is a second at most, so a
# couple of seconds is generous; the point is that a *transient* hold never fails a launch,
# while a genuinely stuck one still reports itself instead of hanging forever.
_SOURCE_WRITE_TIMEOUT: float = 5.0


_SOURCE_WRITE_POLL: float = 0.05


# How many times `_ensure_directory` retries a create another writer is racing.



def _ensure_directory(directory:str)->None:
    """Make `directory` exist; the tolerant implementation lives in `utils`."""
    from ...utils import _ensure_directory as ensure

    ensure(directory)



def _path_lock(path:str)->threading.RLock:
    """The process-wide lock for one generated file, created on first ask."""
    key: str = os.path.normcase(os.path.abspath(path))
    with _WRITE_LOCKS_GUARD:
        lock = _WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _WRITE_LOCKS[key] = lock
    return lock



def _file_lock_path(path:str)->str:
    """Where the cross-process lock for one generated file lives.

    Beside the file it guards, with a `.lock` suffix. The file is created on first use and
    then **kept**: ownership is the OS lock on its handle, not the file's existence (see
    `_file_lock`), so an empty leftover file is not a held lock and nothing has to clean it
    up after a crash.
    """
    return path + ".lock"



def _try_lock(handle:io.IOBase)->bool:
    """Take the OS exclusive lock on an open handle, without waiting; answer whether it was.

    One call, two platforms, and the same meaning on both: an advisory exclusive lock that
    the kernel releases when the handle closes -- which is what happens when the process
    exits *or is killed*, with no cleanup code of ours involved.

    It never blocks, and that is deliberate. Windows' blocking mode retries on a one-second
    tick of its own, and POSIX's would wait inside the C call; either way the caller could
    not honour its own `_FILE_LOCK_POLL` or its deadline. The caller polls this instead.

    Windows has `msvcrt.locking`, which locks a byte range; POSIX has `fcntl.lockf`, whose
    `LOCK_NB` flag turns the wait into `BlockingIOError`. The failure both share is
    `OSError`, and a caller cannot distinguish "held by someone else" from "this filesystem
    does not support locking" by the exception type, so both answer False and the caller
    retries until its deadline.
    """
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True



def _unlock(handle:io.IOBase)->None:
    """Release the OS lock on an open handle, tolerating a handle that never held it."""
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.lockf(handle, fcntl.LOCK_UN)
    except OSError:
        # The handle is closed straight after this, which releases the lock anyway; an
        # unlock that fails because the lock was never taken is not an error here.
        pass



@contextlib.contextmanager
def _file_lock(path:str)->Iterator[None]:
    """Hold the cross-process lock for one generated file.

    Two processes compiling different kernels of the same python module write the same
    generated file, and the write is a read-modify-write: without this, the process that
    reads first and writes last drops the definitions the other one added.

    **Ownership is an OS advisory lock on an open handle, not the file's existence.** The
    lock file is created once and kept; taking the lock is `msvcrt.locking` on Windows or
    `fcntl.lockf` on POSIX. That distinction is the whole point: the previous version
    created the file with `O_CREAT | O_EXCL` and deleted it on release, so ownership was
    "this file exists" -- and a process killed while holding it left the file behind,
    making every later writer poll for the full `_FILE_LOCK_TIMEOUT`. Ten minutes of
    waiting on a lock whose owner is a corpse is not a failure mode a cache write should
    have. With an advisory lock the kernel drops it when the holder dies, so a killed
    process is indistinguishable from one that released it, and no stale state exists to
    detect, reclaim, or get wrong.

    The wait is still bounded, so a genuinely stuck holder (one alive but wedged) reports
    itself instead of hanging the caller forever.

    A thread that already holds this file's lock re-enters it without taking the OS lock
    again. Advisory locks belong to the *process*, not the thread, and `lockf` on a second
    descriptor would deadlock against the process's own first descriptor; the held set is
    per thread, so a nested acquisition (a helper's write inside its caller's critical
    section) re-enters while another thread still waits.
    """
    key: str = os.path.normcase(os.path.abspath(path))
    held: set = getattr(_file_lock_state, "held", None) or set()
    if key in held:
        # This thread already owns the file's lock; taking it again would wait on itself.
        yield
        return

    lock_path: str = _file_lock_path(path)
    _ensure_directory(os.path.dirname(lock_path) or ".")
    # `a+b` creates the lock file if it is absent and never truncates it. The handle is
    # what owns the lock, so it has to stay alive for the whole critical section.
    handle: io.IOBase = open(lock_path, "a+b")
    try:
        deadline: float = time.monotonic() + _FILE_LOCK_TIMEOUT
        while not _try_lock(handle):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out after {_FILE_LOCK_TIMEOUT:.0f}s waiting for another process "
                    f"to finish writing {path}; if no other process is compiling, the holder "
                    f"is stuck (a killed holder releases this lock by itself)") from None
            time.sleep(_FILE_LOCK_POLL)
        held.add(key)
        _file_lock_state.held = held
        try:
            yield
        finally:
            held.discard(key)
            _unlock(handle)
    finally:
        # Closing the handle releases the lock as well, which is what makes the killed
        # case work; the explicit unlock above is only there for a clean, documented exit.
        handle.close()



@contextlib.contextmanager
def write_lock(path:str)->Iterator[None]:
    """Both the thread lock and the process lock for one generated file.

    The order is fixed -- thread lock first, then the lock file -- so two writes cannot
    deadlock against each other, and a thread that already holds its own lock (a helper's
    write nested inside its caller's) simply re-enters it.
    """
    with _path_lock(path):
        with _file_lock(path):
            yield
