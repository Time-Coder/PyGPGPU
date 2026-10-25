"""11_python_backend.py -- run a @kernel function in the interpreter, with no GPU.

`k["python"]` (aliases `"py"`, `"host"`) is the backend that translates nothing: the
decorated function is executed by CPython, serially, on the CPU. It needs no driver,
no compiler, no cache directory and no device, which makes it useful for testing and
for small inputs -- and it is a reference implementation of the launch semantics:

- a magic-loop kernel (`for i in range(x.shape[0])`, `enumerate`, `np.ndindex`) already
  contains the loop over the elements, so the function is called once, as written;
- a gid-style kernel (`gpu.get_global_id`) is run once per work item of its iteration
  domain, with `gpu.get_global_id` reporting the current index -- the serial
  replacement of "one work item per element". The domain comes from `gpu.work_size(x)`
  or from a `(total, local)` bracket.

`submit` / `async_call` work here too, on a ThreadPoolExecutor (one launch is still
serial; concurrent launches must write different arrays).

Run:
    python examples/11_python_backend.py
"""
import asyncio

import numpy as np

from pygpgpu import kernel, gpu


@kernel
def saxpy(x, y, out, alpha):
    """Magic loop: the interpreter runs this once and the loop covers every element."""
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]


@kernel
def histogram(data, counts):
    """gid-style: the interpreter runs this once per index of data (gpu.work_size)."""
    gpu.work_size(data)
    i = gpu.get_global_id(0)
    gpu.atomic_add(counts, data[i], 1)


def main():
    n = 100_000
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    expect = x * 2.0 + y

    # 1) a magic-loop kernel: one plain serial call, no translation at all
    out = np.zeros_like(x)
    saxpy["python"](x, y, out, 2.0)
    assert np.allclose(out, expect)
    print("saxpy['python']            OK")

    # 2) submit -> concurrent.futures.Future (executed on a thread pool)
    out2 = np.zeros_like(x)
    future = saxpy["python"].submit(x, y, out2, 2.0)
    future.result()
    assert np.allclose(out2, expect)
    print("saxpy['python'].submit     OK")

    # 3) async_call -> asyncio.Future
    async def run():
        out3 = np.zeros_like(x)
        await saxpy["python"].async_call(x, y, out3, 2.0)
        return bool(np.allclose(out3, expect))

    print("saxpy['python'].async_call", "OK" if asyncio.run(run()) else "FAILED")

    # 4) gid-style kernel: the declaration supplies the domain, so the body runs once
    #    per element and the atomics add up exactly (serial execution, no races)
    data = np.random.randint(0, 4, size=20_000, dtype=np.int32)
    counts = np.zeros(4, dtype=np.int32)
    histogram["python"](data, counts)
    assert counts.tolist() == np.bincount(data, minlength=4).tolist(), counts
    print("histogram['python']        OK:", counts.tolist())

    # 5) the same function on a GPU backend: uncomment where one is installed
    # saxpy["opencl"](x, y, out, 2.0)
    # saxpy["cuda"](x, y, out, 2.0)


if __name__ == "__main__":
    main()
