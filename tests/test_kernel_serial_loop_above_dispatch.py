"""A serial `for` above the dispatch is per-work-item code, not host code.

`body_tail` used to take the **last `ast.For` of any kind** as the boundary between device
code and the host tail, and `_outside_loops` returned the whole body when it saw no dispatch
loop at all. Both were wrong for a gid-style kernel that carries an ordinary loop above its
declaration:

    for j in range(lane * 4, lane * 4 + 4):     # serial, per work item
        acc += x[j]
    buf = local_array_per_group(256, np.int32)        # device-only builtin

The loop was mistaken for the dispatch, so everything after it -- the shared-memory
declaration included -- became the host *tail*, and `local_array_per_group` then ran on the host
with no work item: "only available inside a @kernel function body", for a body that plainly
was one. The two predicates now agree: a `range` is the dispatch only when it addresses
`x.shape[0]`, which is the shape the translator itself treats as the work-item domain.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import barrier, blockIdx, get_global_id, kernel, local_array_per_group

ITEMS = 1024
LANES = 256


@kernel
def shared_sum_with_serial_loop(x: np.ndarray, out: np.ndarray):
    """Each work item sums its own four elements, then one lane adds the group's slots."""
    lane = get_global_id(0)

    acc = 0
    for j in range(lane * 4, lane * 4 + 4):
        acc += x[j]

    # A literal shape: the buffer is a fixed-size __shared__/__local array.
    buf = local_array_per_group(256, np.int32)
    buf[blockIdx][lane] = acc
    barrier()

    if lane == 0:
        total = 0
        for k in range(256):
            total += buf[blockIdx][k]
        out[0] = total


def test_a_serial_loop_above_the_dispatch_keeps_the_body_on_the_device(gpu_backend):
    """`local_array_per_group` must reach the device, not the host tail."""
    x = np.arange(1, ITEMS + 1, dtype=np.int32)
    out = np.zeros(1, dtype=np.int32)
    shared_sum_with_serial_loop[gpu_backend][(1, LANES)](x, out)
    assert int(out[0]) == int(x.sum()), (int(out[0]), int(x.sum()))


def test_the_predicate_separates_a_serial_range_from_the_dispatch():
    """`range(lane * 4, ...)` is serial; `range(x.shape[0])` is the work-item domain."""
    import ast

    from pygpgpu.kernel.translator import _is_magic_loop_stmt

    serial = ast.parse("for j in range(lane * 4, lane * 4 + 4):\n    pass\n").body[0]
    literal = ast.parse("for k in range(256):\n    pass\n").body[0]
    dispatch = ast.parse("for i in range(x.shape[0]):\n    pass\n").body[0]
    assert not _is_magic_loop_stmt(serial)
    assert not _is_magic_loop_stmt(literal)
    assert _is_magic_loop_stmt(dispatch)
