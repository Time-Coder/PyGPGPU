# The locks around a generated file, the timings they use, and the directory they need.
from __future__ import annotations
import contextlib
import os
import threading
import time
from typing import Dict, Iterator, Optional



# One generated file belongs to one python module, and several threads (and several
# processes) can compile different kernels of that module at the same time. The write is a
# read-modify-write -- read the file, merge the definitions this call rendered into it, and
# write it back -- so without a lock the later writer drops what the earlier one added, and
# the kernel that lost its definition fails to load with a bare `KeyError` naming the entry
# point it cannot find. Two locks are needed and neither replaces the other:
#
# * a process-wide lock per file (the table below), because a `threading.Lock` is what makes
#   this atomic between threads, and threads in one process share every file;
# * an OS-level lock file beside the target (see `_file_lock`), because separate processes
#   share nothing in memory -- the `__clcache__` directory is on disk and every process
#   writing the same module writes the same path.
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

    Beside the file it guards, with a `.lock` suffix: a lock file that outlives a crash
    carries no state to get wrong, because ownership is the OS handle itself -- a killed
    process releases it, so a stale file is an empty file nobody holds, not a deadlock.
    """
    return path + ".lock"



@contextlib.contextmanager
def _file_lock(path:str)->Iterator[None]:
    """Hold the cross-process lock for one generated file.

    Two processes compiling different kernels of the same python module write the same
    generated file, and the write is a read-modify-write: without this, the process that
    reads first and writes last drops the definitions the other one added. The lock is an
    OS-level exclusive create (`O_CREAT | O_EXCL`) on a `.lock` file beside the target,
    which is atomic on every platform this package targets and needs no library.

    The wait is bounded rather than infinite, so a genuinely stuck holder reports itself
    instead of hanging the caller forever. A lock file left behind by a killed process is
    empty and unheld, so an attempt that finds one simply retries and then takes it: the
    ownership is the open handle, not the file's existence.

    A thread that already holds this file's lock re-enters it without taking the lock file
    again. An exclusive create has no notion of "already mine", so a nested acquisition --
    a helper's write inside its caller's critical section, or the write-and-compile pair
    around a kernel that reaches a helper -- would wait on a lock file this very thread is
    holding, which is a deadlock rather than contention. The held set is per thread, so no
    other thread is mistaken for the owner.
    """
    key: str = os.path.normcase(os.path.abspath(path))
    held: set = getattr(_file_lock_state, "held", None) or set()
    if key in held:
        # This thread already owns the file's lock; taking it again would wait on itself.
        yield
        return

    lock_path: str = _file_lock_path(path)
    _ensure_directory(os.path.dirname(lock_path) or ".")
    deadline: float = time.monotonic() + _FILE_LOCK_TIMEOUT
    handle: Optional[int] = None
    while True:
        try:
            # `O_EXCL` makes "create if absent" one atomic step, which is what "I own this
            # file now" means here. Every later writer sees EEXIST and waits.
            handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out after {_FILE_LOCK_TIMEOUT:.0f}s waiting for another process "
                    f"to finish writing {path}; if no other process is compiling, delete "
                    f"{lock_path} and retry") from None
            time.sleep(_FILE_LOCK_POLL)
        except PermissionError:
            # The window between the holder's `os.close` and its `os.unlink` is the one race
            # this file can lose: the lock file still exists but nobody owns it, and on
            # Windows a create that lands inside that window fails with `PermissionError`
            # (a sharing violation on the path being deleted) rather than
            # `FileExistsError`. It is transient by construction -- the holder's unlink
            # finishes in microseconds -- so waiting and retrying is the whole fix; treating
            # it as fatal was the intermittent "Permission denied: ... .cl.lock" the
            # four-process compile test caught about one run in twelve.
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out after {_FILE_LOCK_TIMEOUT:.0f}s waiting for the lock file "
                    f"{lock_path} to be released; if no other process is compiling, delete "
                    f"{lock_path} and retry") from None
            time.sleep(_FILE_LOCK_POLL)
    held.add(key)
    _file_lock_state.held = held
    try:
        yield
    finally:
        held.discard(key)
        if handle is not None:
            os.close(handle)
        with contextlib.suppress(OSError):
            os.unlink(lock_path)



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
