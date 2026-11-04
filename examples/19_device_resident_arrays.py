"""Device-resident arrays (`pygpgpu.numpy`): stay on the GPU across many launches.

An ordinary numpy array travels to the device and back on every kernel call. A `gnp`
array does not: the bytes stay where the kernel left them, and they only come home when
the host actually reads them. This example walks the whole surface -- construction,
residency, explicit transfers, cross-device staging, zero-copy and torch interop.

Run: ``python examples/19_device_resident_arrays.py``
"""
import numpy as np
import pygpgpu
import pygpgpu.numpy as gnp
from pygpgpu import kernel, list_devices

pygpgpu.set_print_info(False)


@kernel
def saxpy(a: np.ndarray, b: np.ndarray, out: np.ndarray, scale: np.float32):
    for i in range(out.shape[0]):
        out[i] = a[i] * scale + b[i]


def title(text):
    print()
    print(text)
    print("-" * len(text))


print("device-resident arrays: pygpgpu.numpy")
print("devices:", ", ".join(list_devices()))

n = 1024
backend = "cuda" if "cuda:0" in list_devices() else (
    "hip" if "hip:0" in list_devices() else "opencl")

# ---------------------------------------------------------------- 1. construction
title("1. Constructing a device-resident array")

a = gnp.arange(n, dtype=np.float32)          # every gnp constructor returns gnp
b = gnp.ones(n, dtype=np.float32)
out = gnp.zeros(n, dtype=np.float32)
print("type:", type(out).__module__ + "." + type(out).__name__)
print("gnp constructors:", ", ".join(
    name for name in ("zeros", "ones", "empty", "full", "arange", "linspace")
    if hasattr(gnp, name)))

# `np.zeros(...)` is numpy's own function, so it stays a plain array with no residency.
print("np.zeros is a plain ndarray:", type(np.zeros(4)).__name__)
print("gnp.zeros is device-resident:", type(out).__name__)

# ---------------------------------------------------------------- 2. residency
title("2. The bytes stay on the device between launches")

# Three launches in a row. With a plain numpy array each of them would copy `out` to the
# device and back; here the array never leaves it.
saxpy[backend](a, b, out, 1.0)
saxpy[backend](out, b, out, 2.0)          # read and write the same buffer on device
saxpy[backend](out, b, out, 3.0)

# Only this read pulls anything back to the host.
print("result[0:4]:", out[0:4].tolist())
print("expected  :", [10.0, 16.0, 22.0, 28.0])

# ---------------------------------------------------------------- 3. public transfers
title("3. `to_host` / `to_device` are public, asynchronous, and return a Transfer")

transfer = out.to_host()
print("Transfer fields:", ", ".join(
    f for f in ("done", "completed", "wait", "result", "future") if hasattr(transfer, f)))

# It is asynchronous: the call returns before the copy has finished. Nothing is lost
# by not waiting here -- the next host read waits for it by itself.
print("bytes identical without an explicit wait:", np.array_equal(
    out.tolist(), [6.0 * i + 10.0 for i in range(n)]))

# Waiting is only needed when you want a hard guarantee at a point of real dependency.
out.to_host().wait()

# The same call spelled with an explicit target, and with `wait=True` for a synchronous
# transfer whose completion you want stated rather than implied.
out.to_device(backend + ":0", wait=True)
print("resident on", backend + ":0")

# ---------------------------------------------------------------- 4. views
title("4. Views are kernel arguments too")

grid = gnp.zeros((4, 4), dtype=np.float32)
grid[0] = 1.0                      # a contiguous row: a window into its parent
row = grid[0]
saxpy[backend](row, row, row, 2.0)
print("row after kernel:", row.tolist(), "(1.0 * 2 + 1.0 = 3.0)")
print("the parent sees it too:", grid[0].tolist())

col = grid[:, 1]                   # a strided view: staged as its own contiguous bytes
print("strided view shape:", col.shape, "contiguous:", bool(col.flags["C_CONTIGUOUS"]))
print("view and parent still agree:", float(grid[0, 1]) == float(col[0]))

# ---------------------------------------------------------------- 5. cross-device
title("5. Copying between devices stages through the host")

others = [name for name in ("cuda:0", "hip:0", "opencl:0")
          if name in list_devices() and name.split(":")[0] != backend]
if others:
    other = others[0]
    # The newest content is what gets moved, so a chain of writes on several devices
    # always leaves the most recent result readable from the host.
    out.to_device(other, wait=True)
    saxpy[other](out, b, out, 2.0)
    print(f"written on {backend}, then read and updated on {other}")
    print("newest value visible from the host:", out[0])
else:
    print("this machine has only one backend, so there is nothing to stage between")
