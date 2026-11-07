"""The async launch of a tail-returning kernel must evaluate the tail after the
future settles. This pins the mechanism the fix installs: _with_returns runs the
launch's tail region (and the outer write-back) inside settle, so a `return first`
after the loop answers the same value submit() does synchronously."""
import numpy as np
from pygpgpu import kernel


@kernel
def first_doubled(x: np.ndarray):
    first = 0.0
    for i in range(x.shape[0]):
        x[i] = x[i] * 2.0
        if i == 0:
            first += x[i]
    return first


def test_submit_returns_the_tail_value(gpu_backend):
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    target = first_doubled[gpu_backend][(2, 4)]
    assert float(target.submit(x).result(timeout=120)) == 2.0
    assert float(x[0]) == 2.0


def test_async_call_returns_the_tail_value(gpu_backend):
    import asyncio

    x = np.array([2.0, 4, 6, 8], dtype=np.float32)
    target = first_doubled[gpu_backend][(2, 2)]

    async def launch():
        return await target.async_call(x)

    assert float(asyncio.run(launch())) == 4.0
    assert float(x[0]) == 4.0
