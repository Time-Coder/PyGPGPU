"""05_gnp_zero_copy_and_atomics.py -- gnp device-resident arrays + atomics.

`pygpgpu.numpy` (aliased gnp) subclasses numpy arrays so the data can stay on
the device:
- it is uploaded the first time it is passed to a kernel;
- consecutive kernel calls no longer shuttle the data back and forth (zero
  intermediate copies);
- the data is read back only when host code actually reads the array contents;
- read-only arguments skip the readback and write-only arguments skip the
  upload (handled automatically by the static read/write analysis).

Device builtin namespace `gpu.*`:
- `atomic_add(arr, idx, val)`: atomic integer add, handy for histograms
  and reductions;
- `local_array_per_group(shape, dtype)`: work-group shared memory, one
  slice per group, indexed by `blockIdx` (`buf[blockIdx][lane]`) +
  `barrier()` for synchronization. The gid-only `local_array(shape, dtype)`
  builds the same storage for a body that is one work item, so it indexes
  directly (`buf[lane]`);
- plus the indexing builtins get_global_id / get_local_id / get_group_id.

Run:
    python examples/05_gnp_zero_copy_and_atomics.py
"""
import numpy as np

import pygpgpu.numpy as gnp
from pygpgpu import (kernel, atomic_add, barrier, blockIdx, get_global_id, get_group_id,
                     get_local_id, local_array_per_group)


@kernel
def saxpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, alpha: float):
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]


@kernel
def add_inplace(x: np.ndarray, c: float):
    for i in range(x.shape[0]):
        x[i] = x[i] + c


@kernel(domain="data")
def histogram(data: np.ndarray, counts: np.ndarray):
    # A gid-style kernel must declare its iteration domain explicitly (there is no magic
    # loop to read it from): `domain="data"` says `data` supplies the launch size,
    # dimension by dimension.
    i = get_global_id(0)
    atomic_add(counts, data[i], 1)


GROUP_ITEMS = 1024  # elements per work group = 4 elements/thread * 256 threads


@kernel
def group_reduce(x: np.ndarray, out: np.ndarray):
    """Reduction: each work group first reduces into local memory, then thread
    0 atomically adds the partial sum into the global out.

    This one is launched with an explicit bracket, so it declares no domain.
    """
    g = get_group_id(0)
    l = get_local_id(0)
    base = g * GROUP_ITEMS

    # Each thread accumulates the 4 elements it owns
    s = 0
    for j in range(l * 4, l * 4 + 4):
        s += x[base + j]

    buf = local_array_per_group(256, np.int32)  # shared memory, one slice per work group
    buf[blockIdx][l] = s
    barrier()  # the whole group must finish writing before anyone reads

    if l == 0:
        total = 0
        for k in range(256):
            total += buf[blockIdx][k]
        atomic_add(out, 0, total)


def main():
    # --- gnp zero-copy ---
    n = 100_000
    a = gnp.arange(n, dtype=np.float32)
    b = gnp.arange(n, dtype=np.float32)
    out = gnp.empty(n, dtype=np.float32)

    saxpy(a, b, out, 2.0)
    saxpy(a, b, out, 2.0)  # twice in a row: after the first upload the data stays on the device
    # Readback happens only when host code reads out[i]; printing a couple of
    # elements does not slow down the comparison.
    print("saxpy out[0],out[1] =", float(out[0]), float(out[1]))

    # In-place update of a (read/write decided by static analysis; here a is
    # read and written, so it stays resident on the device)
    add_inplace(a, 1.0)
    print("a[0] after +1 =", float(a[0]))

    # --- Atomic histogram ---
    data = np.array([0, 1, 2, 1, 2, 2, 0, 3], dtype=np.int32)
    counts = np.zeros(4, dtype=np.int32)
    histogram(data, counts)
    assert counts.tolist() == [2, 2, 3, 1], counts
    print("Histogram OK:", counts.tolist())

    # --- Local-memory reduction ---
    # The element count must be a multiple of GROUP_ITEMS
    items = 4 * GROUP_ITEMS  # 4 groups
    x = np.arange(1, items + 1, dtype=np.int32)
    total = np.zeros(1, dtype=np.int32)
    # The bracket is (blocks_per_grid, threads_per_block), so the grid is one block per
    # GROUP_ITEMS and each block reads its own 1024 elements: `items` here would be
    # 4096 blocks of 256 threads, four times the domain, and the extra blocks would
    # index past the end of `x`.
    group_reduce[(items // GROUP_ITEMS, 256)](x, total)
    assert total[0] == np.arange(1, items + 1).sum(), (total[0], x.sum())
    print("Local-memory reduction OK: sum =", total.tolist())


if __name__ == "__main__":
    main()