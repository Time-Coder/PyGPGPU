"""09_reduction.py -- array reduction: collapsing many elements into one value.

A reduction cannot be written like an elementwise kernel. Many work items write the
same output slot, and `out[0] = out[0] + x[i]` is a non-atomic read-modify-write, so
the result would be wrong. This example builds up the three usual solutions, from
simplest to fastest:

1. `sum_atomic`      -- every work item adds one element with an atomic add. Simple
                        and always correct, but one atomic per element, so contention
                        grows with the array.
2. `sum_strided`     -- each work item walks a strided slice (`i, i+stride, ...`) and
                        keeps a private sum in a register, then does a single atomic.
                        Only one atomic per work item instead of per element.
3. `sum_local_tree`  -- each work group reduces its per-work-item sums through local
                        (shared) memory with a tree, so a group needs log2(group)
                        steps instead of one atomic per work item.

A gid-style kernel (`get_global_id`) has no loop for the automatic work size to read
the iteration domain from, so each one opens with `work_size(x)`. That statement is
host-side only -- it is erased at translation, emits no device code, and tells the
launcher that `x`'s full shape is the domain. Without it, the call raises rather than
silently launching a one-work-item grid. A bracketed launch such as
`k[(total, local)](...)` also works, and the total must be a multiple of the per-group
size.

The output slot is `int32` because `atomic_add` only supports integer element
types. Two ways to sum floats: scale the data to integers, or use the local-memory
tree and let a single work item add the float partial sums (one writer, so a plain
add is safe there).

Run:
    python examples/09_reduction.py
"""
import numpy as np

from pygpgpu import (kernel, atomic_add, barrier, blockIdx, get_global_id,
                     get_global_size, get_group_id, get_local_id, get_local_size, zeros_per_group,
                     work_size)


def launch_size(n, local=256):
    """Round a work-item count up to a multiple of the per-group size."""
    return ((n + local - 1) // local) * local


@kernel
def sum_atomic(x: np.ndarray, out: np.ndarray):
    """One atomic add per element: the simplest correct reduction."""
    work_size(x)  # declare the iteration domain; erased at translation
    i = get_global_id(0)
    atomic_add(out, 0, x[i])


@kernel
def sum_strided(x: np.ndarray, out: np.ndarray, n: np.int32):
    """Each work item sums a strided slice into a register, then one atomic add.

    `range()` steps must be constant in a kernel, so the strided walk uses `while`.
    The bound is `n`, the real element count: the launch size may be rounded up past
    it, so the extra work items must not read out of range.
    """
    work_size(x)
    i = get_global_id(0)
    stride = get_global_size(0)

    s = 0
    j = i
    while j < n:
        s += x[j]
        j += stride

    atomic_add(out, 0, s)


GROUP = 256  # work items per group, matching the local buffer size below


@kernel
def sum_local_tree(x: np.ndarray, out: np.ndarray, n: np.int32):
    """Tree reduction through local (shared) memory: log2(GROUP) steps per group."""
    work_size(x)
    g = get_group_id(0)
    l = get_local_id(0)
    group_size = get_local_size(0)
    stride = get_global_size(0)

    # 1) Every work item reduces its strided slice privately.
    s = 0
    j = g * group_size + l
    while j < n:
        s += x[j]
        j += stride

    # `zeros_per_group` needs a literal count, so this mirrors GROUP.
    buf = zeros_per_group(256, np.int32)      # shared memory, one slice per work group
    buf[blockIdx][l] = s
    barrier()  # the whole group must finish writing before anyone reads

    # 2) Halve the active range each step: 256 -> 128 -> ... -> 1.
    offset = group_size // 2
    while offset > 0:
        if l < offset:
            buf[blockIdx][l] = buf[blockIdx][l] + buf[blockIdx][l + offset]
        barrier()
        offset = offset // 2

    # 3) Work item 0 of each group adds the group total to the global output.
    if l == 0:
        atomic_add(out, 0, buf[blockIdx][0])


def main():
    n = 100_000
    x = np.random.randint(0, 100, size=n, dtype=np.int32)
    expected = int(x.sum(dtype=np.int64))
    print(f"sum of {n} int32 values, expected = {expected}")

    # 1) One atomic per element -- work size from the work_size(x) declaration
    out = np.zeros(1, dtype=np.int32)
    sum_atomic(x, out)
    assert int(out[0]) == expected, (int(out[0]), expected)
    print("  sum_atomic      OK:", int(out[0]))

    # 2) Private strided sums, then one atomic per work item.
    # Here the launch size is given explicitly instead, to show both styles; note that
    # (total, local) is the same (total-work-item) bracketing as 01_basic_kernel.py.
    out = np.zeros(1, dtype=np.int32)
    sum_strided[(launch_size(n), 256)](x, out, np.int32(n))
    assert int(out[0]) == expected, (int(out[0]), expected)
    print("  sum_strided     OK:", int(out[0]))

    # 3) Tree reduction in local memory -- automatic, from the declaration
    out = np.zeros(1, dtype=np.int32)
    sum_local_tree(x, out, np.int32(n))
    assert int(out[0]) == expected, (int(out[0]), expected)
    print("  sum_local_tree  OK:", int(out[0]))

    # --- awkward lengths ---
    # The launch size is rounded up to a multiple of the local size, so these lengths
    # exercise the clamp: the extra work items must contribute nothing.
    for length in (997, 65_537):
        y = np.random.randint(0, 100, size=length, dtype=np.int32)
        want = int(y.sum(dtype=np.int64))
        for name, call in (
            ("sum_strided   ", lambda o: sum_strided(y, o, np.int32(length))),
            ("sum_local_tree", lambda o: sum_local_tree(y, o, np.int32(length))),
        ):
            got = np.zeros(1, dtype=np.int32)
            call(got)
            assert int(got[0]) == want, (name, int(got[0]), want)
            print(f"  {name} length {length}: OK ({want})")

    # --- float data ---
    # atomic_add is integer-only. For floats, reduce into local memory and let one
    # work item add the partials: a single writer, so a plain add is safe.
    print("  float sum: see the note in this module's docstring")
    f = np.arange(1, 4097, dtype=np.float32)
    assert np.isclose(float(f.sum()), 4096 * 4097 / 2)


if __name__ == "__main__":
    main()
