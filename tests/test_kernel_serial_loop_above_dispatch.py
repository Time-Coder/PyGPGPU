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

    params = {"x", "lane"}
    serial = ast.parse("for j in range(lane * 4, lane * 4 + 4):\n    pass\n").body[0]
    literal = ast.parse("for k in range(256):\n    pass\n").body[0]
    dispatch = ast.parse("for i in range(x.shape[0]):\n    pass\n").body[0]
    assert not _is_magic_loop_stmt(serial, params)
    assert not _is_magic_loop_stmt(literal, params)
    assert _is_magic_loop_stmt(dispatch, params)


def test_a_runtime_count_is_a_dispatch_and_a_constant_is_not():
    """`range(n)` over a parameter is a domain; `range(8)` is a walk one work item does.

    The line is "the call decides the count" versus "the source decides it": a parameter is a
    value only the caller knows, so its loop is one iteration per work item; a literal or a
    module constant is a property of the source, so its loop is a fixed number of steps inside
    one work item. Promoting the second would turn every inner walk (a raymarcher's march, a
    matmul's row dot) into a grid.
    """
    import ast

    from pygpgpu.kernel.translator import _is_magic_loop_stmt

    params = {"n", "x"}
    counted = ast.parse("for i in range(n):\n    pass\n").body[0]
    literal = ast.parse("for i in range(8):\n    pass\n").body[0]
    local = ast.parse("for i in range(k):\n    pass\n").body[0]   # `k` is not a parameter
    stepped = ast.parse("for i in range(0, n, 2):\n    pass\n").body[0]
    assert _is_magic_loop_stmt(counted, params)
    assert not _is_magic_loop_stmt(literal, params)
    assert not _is_magic_loop_stmt(local, params)
    # A start/step form is a strided walk, not a one-iteration-per-item domain.
    assert not _is_magic_loop_stmt(stepped, params)


def test_a_counted_domain_without_the_parameter_set_stays_serial():
    """The parameter set is what makes a counted domain recognisable, and it is required.

    A caller that could omit it would silently get the shape-only answer, and then the
    host-region boundary and the translator would disagree about which loop is the
    dispatch -- the disagreement this signature is written to make impossible.
    """
    import ast

    from pygpgpu.kernel.translator import _is_magic_loop_stmt

    counted = ast.parse("for i in range(n):\n    pass\n").body[0]
    assert not _is_magic_loop_stmt(counted, frozenset())
