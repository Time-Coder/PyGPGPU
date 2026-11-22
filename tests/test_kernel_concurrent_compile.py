"""Concurrent compilation and loading of one module's kernels.

A python module owns one generated device file, and every kernel it defines is appended to
that same file. Two threads (or two processes) that compile *different* kernels of that
module therefore touch the same path, and the failure that used to follow was not a
corruption but a missing symbol: the later writer's read-modify-write dropped the earlier
one's definition, so the launch reported `KeyError` naming the kernel it could not find.

Four independent things had to be serialized, and each is pinned here:

* the generated source's read-merge-write (`source_paths.write_lock`);
* the compile that follows it, because the cached binary is keyed on the file's *content*
  -- a compile starting between two appends cached a binary of a half-populated file;
* the lazy device and platform enumeration, which returned `CL_DEVICE_NOT_FOUND` when two
  threads ran it at once rather than merely repeating the work;
* the binary and meta cache writes, which are atomic now so a reader never loads a
  truncated artifact (`CL_INVALID_KERNEL`).

The tests are deliberately about *observable* correctness: every thread checks the
arithmetic it asked for, so a silently wrong launch fails as loudly as an exception.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import threading
import time
import traceback

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.kernel import source_paths


BACKEND = "opencl"

OPERANDS = (6.0, 3.0)


@kernel
def concurrent_add(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


@kernel
def concurrent_mul(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * b[i]


@kernel
def concurrent_sub(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] - b[i]


@kernel
def concurrent_div(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] / b[i]


# Each kernel answers a different value from the same inputs, so a launch that returned
# another kernel's result is caught by the comparison rather than passing silently.
KERNELS = (concurrent_add, concurrent_mul, concurrent_sub, concurrent_div)
EXPECTED = (9.0, 18.0, 3.0, 2.0)


def _clear_cache() -> None:
    """Delete this module's generated source, so the next call compiles from scratch.

    The contention only exists while the file is being built, so each test starts from an
    empty cache; a warm module writes nothing and the locks are not exercised.
    """
    root = source_paths.cache_root(__file__, BACKEND)
    shutil.rmtree(root, ignore_errors=True)


def _call(index: int) -> str | None:
    """Launch one kernel and report what went wrong, or None when it was correct."""
    k = KERNELS[index % len(KERNELS)]
    a = np.full(8, OPERANDS[0], dtype=np.float32)
    b = np.full(8, OPERANDS[1], dtype=np.float32)
    out = np.zeros(8, dtype=np.float32)
    try:
        k[BACKEND](a, b, out)
    except BaseException:
        return f"[{k.__name__}] " + traceback.format_exc().strip().splitlines()[-1]
    expected = EXPECTED[index % len(KERNELS)]
    if not np.allclose(out, expected):
        return f"[{k.__name__}] wrong result {out.tolist()}"
    return None


def _call_first(_ignored: object) -> str | None:
    """A picklable entry point: `spawn` cannot pass a plain function to a pool."""
    return _call(0)


def _hold_the_lock(source: str, ready: object) -> None:
    """Take the source's write lock, announce it, then wait to be killed.

    Exiting this function normally would release the lock, so it spins instead and leaves
    the release to the process being killed -- which is the case the test is about.
    """
    with source_paths.write_lock(source):
        ready.set()  # type: ignore[attr-defined]
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            time.sleep(0.05)


@pytest.fixture
def cold_cache() -> None:
    _clear_cache()


def test_concurrent_threads_compile_distinct_kernels(cold_cache, gpu_backend):
    """Fifteen threads over four kernels: every launch answers its own arithmetic."""
    failures: list[str | None] = [None] * 15

    def worker(index: int) -> None:
        failures[index] = _call(index)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(15)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    errors = [failure for failure in failures if failure]
    assert not errors, errors[:3]


def test_concurrent_processes_compile_the_same_kernel(cold_cache, gpu_backend):
    """Separate processes must serialize their writes through the lock file alone.

    Nothing in memory is shared between them, so this is the case the `.lock` file exists
    for: without it both processes read the empty file, both merge only their own
    definition, and the later write erases the earlier one.
    """
    with mp.get_context("spawn").Pool(4) as pool:
        results = pool.map(_call_first, [None] * 4)
    errors = [result for result in results if result]
    assert not errors, errors[:3]


def test_releasing_the_lock_leaves_no_ownership_behind(cold_cache):
    """The lock file stays; the *lock* does not.

    Ownership is an OS advisory lock on an open handle, so the file itself is created once
    and kept -- an empty `.lock` beside the source is a normal, unheld file, not a stale
    lock. What has to be gone is the exclusion, which is what this checks: after the block
    exits, another acquire takes it at once rather than waiting out the timeout.
    """
    source = source_paths.source_path(__file__, __file__, BACKEND)
    with source_paths.write_lock(source):
        assert os.path.isfile(source + ".lock")

    started = time.monotonic()
    with source_paths.write_lock(source):
        pass
    assert time.monotonic() - started < 5.0, "a released lock must be takeable at once"


def test_a_killed_holders_lock_is_released(cold_cache):
    """A process killed while holding the lock does not block the next one.

    Ownership is an OS advisory lock, so the kernel drops it when the holder dies: the cache
    write that follows a cancelled test, a debugger stop or a `kill -9` proceeds at once.

    The alternative design -- an exclusively *created* lock file, deleted on release -- fails
    this: the killed holder leaves the file behind, nobody owns it, and every later writer
    polls until `_FILE_LOCK_TIMEOUT`, ten minutes of waiting on a lock whose owner is a
    corpse. That is exactly what happened before this test existed, so the timeout here is
    shortened and the elapsed time is what is asserted.
    """
    source = source_paths.source_path(__file__, __file__, BACKEND)
    os.makedirs(os.path.dirname(source + ".lock"), exist_ok=True)

    ready = mp.get_context("spawn").Event()
    child = mp.get_context("spawn").Process(target=_hold_the_lock, args=(source, ready))
    child.start()
    try:
        assert ready.wait(60), "the child never took the lock"
        child.kill()
    finally:
        child.join()

    from pygpgpu.kernel.source_paths import locks

    original = locks._FILE_LOCK_TIMEOUT
    locks._FILE_LOCK_TIMEOUT = 5.0
    started = time.monotonic()
    try:
        with source_paths.write_lock(source):
            elapsed = time.monotonic() - started
    finally:
        locks._FILE_LOCK_TIMEOUT = original
    assert elapsed < 5.0, f"waited {elapsed:.1f}s on a killed holder's lock"


def test_the_source_lock_is_reentrant(cold_cache):
    """A helper compiled inside its caller's critical section re-enters, not deadlocks."""
    source = source_paths.source_path(__file__, __file__, BACKEND)
    with source_paths.write_lock(source):
        with source_paths.write_lock(source):
            pass


def test_two_writers_do_not_lose_each_others_definitions(tmp_path):
    """The merge is under one lock, so neither writer's definition disappears."""
    path = str(tmp_path / "shared.cl")
    first = "float first_one(float x)\n{\n    return x;\n}"
    second = "float second_one(float x)\n{\n    return x + 1.0f;\n}"

    def write(piece: str) -> None:
        source_paths.append_source(path, [], [piece], __file__)

    threads = [threading.Thread(target=write, args=(piece,))
               for piece in (first, second, first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    text = open(path, "r", encoding="utf-8").read()
    # Both definitions survive, and neither is duplicated.
    assert text.count("first_one") == 1, text
    assert text.count("second_one") == 1, text
