"""08_compare_backends.py -- speed of one kernel across OpenCL / CUDA / HIP.

Do not hardcode assumptions about the environment: first probe what backends
and how many devices this machine really has with `list_devices()`, and
only run the backends that actually exist. Repeat a few times and report the
timing for every device.

The comparison uses the same kernel and the same data size throughout:
- `perf_counter` wraps the calls made "after the compile is cached" (the first
  compile after a cache invalidation is excluded),
- only the python-side cost of launching the kernel is measured; precise
  device-side timestamps are not included (see the comments).

Run:
    python examples/08_compare_backends.py

Typical output shape:
    cuda:0   NVIDIA ...     12.3 ms
    hip:0    AMD Radeon ... 18.7 ms
    opencl:0 NVIDIA ...     14.1 ms
    opencl:1 AMD gfx1036 ... 22.4 ms
(A missing backend simply leaves out its rows.)
"""
import time
import numpy as np

from pygpgpu import kernel, list_devices, get_device

N = 4 * 1024 * 1024          # data size
REPEATS = 8                  # repeats per device (median is taken)


@kernel
def saxpy(x, y, out, alpha):
    """Kernel used for the comparison: out[i] = x[i] * alpha + y[i]."""
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]


def main():
    sel = list_devices()
    if not sel:
        print("This machine has no usable GPU backend.")
        return

    x = np.random.rand(N).astype(np.float32)
    y = np.random.rand(N).astype(np.float32)
    out = np.empty_like(y)
    alpha = np.float32(2.0)

    print(f"Data size {N:,} float32; {REPEATS} repeats per device, median reported\n")
    print(f"{'Backend:Dev':<14s} {'Device name':40s} {'Median time':>12s}")

    rows = []
    for key in sel:
        dev = get_device(key)
        # The first call triggers compilation for this backend/device (including
        # the cache check) and is not timed; the following REPEATS calls all hit
        # the compile cache, so the timing reflects run overhead only.
        saxpy[key](x, y, out, alpha)

        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            saxpy[key](x, y, out, alpha)
            times.append((time.perf_counter() - t0) * 1e3)
        med = sorted(times)[len(times) // 2]
        rows.append((key, dev.name, med))
        print(f"{key:<14s} {dev.name[:40]:<40s} {med:9.3f} ms")

    # A one-line conclusion: which one is fastest
    if rows:
        best = min(rows, key=lambda r: r[2])
        print(f"\nFastest: {best[0]} ({best[1]}) {best[2]:.3f} ms")


if __name__ == "__main__":
    main()