"""The asynchronous forms of a multi-loop body: every pass runs, in order, off-thread."""
import asyncio
import numpy as np
from pygpgpu import kernel


@kernel
def two_passes_async(out: np.ndarray, n: int):
    """Two dependent passes; the host region between them updates the parameter."""
    for i in range(out.shape[0]):
        out[i] = float(i)
    n = n + 1
    for i in range(out.shape[0]):
        out[i] = out[i] + float(n)


@kernel
def two_passes_tail(x: np.ndarray, n: int):
    """A tail `return` after the last pass; the region updates the parameter."""
    for i in range(x.shape[0]):
        x[i] = x[i] * 2.0
    n = n + 1
    for i in range(x.shape[0]):
        x[i] = x[i] + float(n)
    return float(x[0])


def test_submit_runs_every_pass(gpu_backend):
    """A multi-pass body chained from the backend's completion events, not a thread."""
    out = np.zeros(4, dtype=np.float32)
    future = two_passes_async[gpu_backend].submit(out, 3)
    future.result(timeout=120)
    np.testing.assert_array_equal(out, np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))


def test_submit_runs_the_tail_after_the_last_pass(gpu_backend):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = two_passes_tail[gpu_backend].submit(x, 1).result(timeout=120)
    assert float(result) == 4.0
    np.testing.assert_array_equal(x, np.array([4.0, 6.0, 8.0], dtype=np.float32))


def test_async_call_runs_every_pass(gpu_backend):
    out = np.zeros(4, dtype=np.float32)
    target = two_passes_async[gpu_backend]

    async def launch():
        return await target.async_call(out, 4)

    asyncio.run(launch())
    np.testing.assert_array_equal(out, np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
