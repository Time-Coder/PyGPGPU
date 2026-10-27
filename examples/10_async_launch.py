"""10_async_launch.py -- launch a @kernel function without blocking.

A decorated kernel accepts the same two non-blocking methods as a handwritten
one:

    k.submit(...)      -> concurrent.futures.Future
    k.async_call(...)  -> asyncio.Future

Both return as soon as the launch is enqueued. The output arrays are written
back by the completion callback, so a future must resolve before its results are
read. The bracket selectors work exactly as on a normal call, so
`k["opencl"].submit(...)` picks a backend and `k[(total, local)].submit(...)`
fixes the work size.

This example names the OpenCL backend explicitly so it is deterministic on a
machine with several backends. The asyncio path below is deliberately limited to
one launch at a time; see the README note on asynchronous launching.

Run:
    python examples/10_async_launch.py
"""
import asyncio

import numpy as np

from pygpgpu import kernel


@kernel
def saxpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, alpha: float):
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]


BACKEND = "opencl"


def main():
    n = 1_000_000
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    expect = x * 2.0 + y

    # 1. the blocking call, for reference
    out = np.zeros_like(x)
    saxpy[BACKEND](x, y, out, 2.0)
    print("blocking call      :", bool(np.allclose(out, expect)))

    # 2. concurrent.futures: launch, then wait
    out2 = np.zeros_like(x)
    future = saxpy[BACKEND].submit(x, y, out2, 2.0)
    future.result()
    print("submit             :", bool(np.allclose(out2, expect)))

    # 3. several launches in flight at once
    outs = [np.zeros_like(x) for _ in range(4)]
    futures = [saxpy[BACKEND].submit(x, y, o, 2.0) for o in outs]
    for f in futures:
        f.result()
    print("4 in flight        :", all(bool(np.allclose(o, expect)) for o in outs))

    # 4. asyncio: await the launch instead of blocking on it
    async def run():
        out3 = np.zeros_like(x)
        await saxpy[BACKEND].async_call(x, y, out3, 2.0)
        return bool(np.allclose(out3, expect))

    print("asyncio            :", asyncio.run(run()))


if __name__ == "__main__":
    main()
