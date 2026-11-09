"""A kernel call from a thread that did not create the device context.

`cuCtxSetCurrent` / `hipSetDevice` are per thread: the context a driver call needs is bound
to the thread that created it, and a fresh thread has nothing bound. Every launch path
therefore binds the device's context before its first driver call, which is what lets a
kernel be called from a worker thread at all -- without it the first call fails with
`CUDA_ERROR_INVALID_CONTEXT` (or HIP's equivalent) before anything is enqueued. Already
current is the common case, so the binding costs one query there.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from pygpgpu import kernel


@kernel
def thread_single(a: np.ndarray, b: np.ndarray):
    for i in range(a.shape[0]):
        b[i] = a[i] + 1.0


@kernel
def thread_two_passes(a: np.ndarray, b: np.ndarray):
    for i in range(a.shape[0]):
        a[i] = a[i] * 2.0
    for i in range(a.shape[0]):
        b[i] = a[i] + 1.0


def _in_thread(call):
    """Run `call` on a fresh thread and return its value, re-raising what it raised."""
    box = {}

    def body():
        try:
            box["value"] = call()
        except BaseException as error:  # noqa: BLE001 - reported to the test thread
            box["error"] = error

    worker = threading.Thread(target=body)
    worker.start()
    worker.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_a_launch_from_a_worker_thread_binds_the_context(gpu_backend):
    """The whole point: a plain call from a thread that never launched before."""
    if gpu_backend == "python":
        pytest.skip("the interpreter has no device context to bind")
    values = np.arange(4, dtype=np.float32)
    # Warm the compile on this thread; the worker's first driver call is the launch.
    thread_single[gpu_backend](values, np.zeros(4, dtype=np.float32))

    def call():
        out = np.zeros(4, dtype=np.float32)
        thread_single[gpu_backend](values, out)
        return out

    np.testing.assert_array_equal(_in_thread(call),
                                  np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_a_multi_pass_launch_from_a_worker_thread_binds_the_context(gpu_backend):
    """A multi-pass body syncs before its first pass, so that driver call needs it too."""
    if gpu_backend == "python":
        pytest.skip("the interpreter has no device context to bind")
    thread_two_passes[gpu_backend](np.arange(4, dtype=np.float32),
                                   np.zeros(4, dtype=np.float32))

    def call():
        out = np.zeros(4, dtype=np.float32)
        thread_two_passes[gpu_backend](np.arange(4, dtype=np.float32), out)
        return out

    np.testing.assert_array_equal(_in_thread(call),
                                  np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32))


def test_several_worker_threads_launch_independently(gpu_backend):
    """Each thread binds for itself, so concurrent first launches all work."""
    if gpu_backend == "python":
        pytest.skip("the interpreter has no device context to bind")
    thread_single[gpu_backend](np.arange(4, dtype=np.float32), np.zeros(4, dtype=np.float32))
    results = []
    lock = threading.Lock()
    errors = []

    def body(offset):
        try:
            out = np.zeros(4, dtype=np.float32)
            thread_single[gpu_backend](np.arange(4, dtype=np.float32) + offset, out)
            with lock:
                results.append(out.tolist())
        except BaseException as error:  # noqa: BLE001
            with lock:
                errors.append(error)

    workers = [threading.Thread(target=body, args=(index,)) for index in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if errors:
        raise errors[0]
    assert sorted(results) == [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
    ]


def test_submit_from_a_worker_thread_binds_the_context(gpu_backend):
    """The asynchronous path is a driver call too, so it binds before the first one."""
    if gpu_backend == "python":
        pytest.skip("the interpreter has no device context to bind")
    thread_single[gpu_backend](np.arange(4, dtype=np.float32), np.zeros(4, dtype=np.float32))

    def call():
        out = np.zeros(4, dtype=np.float32)
        future = thread_single[gpu_backend].submit(np.arange(4, dtype=np.float32), out)
        future.result()
        return out

    np.testing.assert_array_equal(_in_thread(call),
                                  np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
