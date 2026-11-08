"""Host regions -- the statements above, between and after a body's magic loops.

A kernel body runs in three regions that are ordinary python: before the first magic loop,
between two of them, and after the last one. The interpreter runs each region **once**,
exactly as the source reads, and the values it binds reach the kernel as parameters. So
anything python can do is legal there -- an import, a dict, a string, an arbitrary-
precision number, a call into a library the device knows nothing about -- as long as what
crosses to the device is a value a device argument can carry.
"""
import numpy as np
import pytest

from pygpgpu import kernel, set_type_map

@pytest.fixture(autouse=True)
def double_float_mapping():
    """A python `float` travels as a double, and the previous default is restored.

    Every value here is a host scalar, so the width has to be wide enough for the python
    float the region computed. The mapping is process-wide, so restoring the module's own
    default afterwards is what keeps the following test modules on theirs.
    """
    from pygpgpu.kernel import decorator  # noqa: PLC0415

    saved = decorator._type_map_default  # noqa: SLF001 (restored in the finally)
    set_type_map(float=np.float64)
    try:
        yield
    finally:
        decorator._type_map_default = saved  # noqa: SLF001 (restored here)


# ---------------------------------------------------------------------------
# Above the first magic loop
# ---------------------------------------------------------------------------


@kernel
def host_string_length(out: np.ndarray):
    """A host-only value (`len` of a string) feeding the work items."""
    label = "hello"
    width = len(label)
    for i in range(out.shape[0]):
        out[i] = i + width


@kernel
def host_import(out: np.ndarray):
    """A module import: it binds a module, not a device value."""
    import json
    text = json.dumps({"n": 4})
    width = len(text)
    for i in range(out.shape[0]):
        out[i] = i + width


@kernel
def host_dict(out: np.ndarray):
    """A dict literal and its length."""
    table = {"a": 2}
    count = len(table)
    for i in range(out.shape[0]):
        out[i] = i + count


@kernel
def host_float_math(out: np.ndarray):
    """Plain python arithmetic over a host-only value."""
    label = "abcd"
    base = len(label) * 1.5
    for i in range(out.shape[0]):
        out[i] = i + base


@kernel
def device_member_above_loop(out: np.ndarray):
    """A cheap scalar above the loop: it is read by the work items."""
    total = 3 + 4
    for i in range(out.shape[0]):
        out[i] = i + total


# ---------------------------------------------------------------------------
# Between two loops, and after the last one
# ---------------------------------------------------------------------------


@kernel
def host_between_loops(first: np.ndarray, second: np.ndarray):
    """A host-only value computed between two passes."""
    for i in range(first.shape[0]):
        first[i] = i * 2.0
    label = "abc"
    shift = len(label)
    for i in range(second.shape[0]):
        second[i] = first[i] + shift


@kernel
def host_tail_reads_a_value(out: np.ndarray):
    """A host-only method after the last loop."""
    for i in range(out.shape[0]):
        out[i] = i


def test_a_host_value_between_two_passes(runner):
    first = np.zeros(4, np.float64)
    second = np.zeros(4, np.float64)
    runner(host_between_loops, first, second)
    np.testing.assert_allclose(first, np.arange(4) * 2.0)
    np.testing.assert_allclose(second, np.arange(4) * 2.0 + 3)


@kernel
def param_updated_between_passes(out: np.ndarray, n: int):
    """A host-region assignment to a kernel parameter, read by the next pass."""
    for i in range(out.shape[0]):
        out[i] = n
    n = n + 1
    for i in range(out.shape[0]):
        out[i] = out[i] + n


def test_a_parameter_assigned_between_two_passes_reaches_the_second(runner):
    """A parameter is python's function-local: the region's `n = n + 1` must reach pass 2.

    The region runs on the host, and the argument list of the next pass was built once, so
    the assignment used to be dropped and the second pass re-read the caller's original
    value (python answers 7, the device answered 6). A parameter is not a gap slot: the
    update belongs to the launch arguments, and the caller's own variable is untouched.
    """
    out = np.zeros(2, np.float32)
    runner(param_updated_between_passes, out, 3)
    np.testing.assert_array_equal(out, np.array([7.0, 7.0], dtype=np.float32))


@kernel
def array_written_between_passes(x: np.ndarray, out: np.ndarray):
    """A region writes into an array argument, and the next pass reads it."""
    for i in range(x.shape[0]):
        x[i] = x[i] * 2.0
    x[0] = x[0] + 1.0
    for i in range(x.shape[0]):
        out[i] = x[i] + 1.0


def test_an_array_write_between_passes_happens_once(runner):
    """The discovery pass runs on copies; the real run happens between the passes.

    `_host_region_values` executes the region once at assembly time to learn which
    statements the host can run, and the launch runs it again at the moment python would.
    On the caller's array the store was therefore applied twice -- and once before the
    first pass, which uploaded the shifted array (python answers [4, 5, 7], the device
    answered [10, 7, 9]).
    """
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(array_written_between_passes, x, out)
    np.testing.assert_array_equal(x, np.array([3.0, 4.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([4.0, 5.0, 7.0], dtype=np.float32))


@kernel
def gap_value_from_the_first_pass(x: np.ndarray, out: np.ndarray):
    """A scalar the region computes from the previous pass's array."""
    for i in range(x.shape[0]):
        x[i] = x[i] * 2.0
    shift = int(x[0])
    for i in range(out.shape[0]):
        out[i] = float(shift)


def test_a_region_value_is_recomputed_after_the_previous_pass(runner):
    """The discovery value and the launch-time value are both bound, and the later wins."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(gap_value_from_the_first_pass, x, out)
    np.testing.assert_array_equal(out, np.array([2.0, 2.0, 2.0], dtype=np.float32))


def test_an_outer_scalar_is_written_by_the_device(runner):
    """`nonlocal held; held = fresh` travels through its own slot, and the device writes it.

    Running the assignment on the host instead would overwrite the slot's initial value
    with the placeholder the region was executed with, and the kernel's answer would be
    lost -- the output was [0, 0] before the host regions learned to leave outer names
    alone.
    """
    held = np.complex128(0 + 0j)

    @kernel
    def rebind(fresh, out: np.ndarray):
        nonlocal held
        held = fresh
        for i in range(out.shape[0]):
            out[i] = held

    out = np.zeros(2, np.complex128)
    runner(rebind, np.complex128(5 + 6j), out)
    np.testing.assert_allclose(out, np.array([5 + 6j, 5 + 6j], dtype=np.complex128))
    assert held == np.complex128(5 + 6j)


def test_zeros_per_group_above_the_loop_is_still_device_code(runner):
    """A device-only builtin has no host meaning, so the host leaves it alone."""
    from pygpgpu import barrier, blockIdx, zeros_per_group

    @kernel
    def shared_above_loop(out: np.ndarray):
        buf = zeros_per_group(4, np.float32)
        for i in range(out.shape[0]):
            buf[blockIdx][0] = 1.0
            barrier()
            out[i] = buf[blockIdx][0]

    result = np.zeros(8, np.float32)
    runner(shared_above_loop, result)
    np.testing.assert_allclose(result, np.ones(8))


def test_a_string_bound_above_the_loop(runner):
    out = np.zeros(4, np.float64)
    runner(host_string_length, out)
    np.testing.assert_allclose(out, np.arange(4) + 5)


def test_an_import_above_the_loop(runner):
    out = np.zeros(4, np.float64)
    runner(host_import, out)
    # json.dumps({"n": 4}) is the 8-character text '{"n": 4}'
    np.testing.assert_allclose(out, np.arange(4) + 8)


def test_a_dict_above_the_loop(runner):
    out = np.zeros(4, np.float64)
    runner(host_dict, out)
    np.testing.assert_allclose(out, np.arange(4) + 1)


def test_float_arithmetic_over_a_host_value(runner):
    out = np.zeros(4, np.float64)
    runner(host_float_math, out)
    np.testing.assert_allclose(out, np.arange(4) + 6.0)


def test_a_plain_scalar_above_the_loop_stays_device_code(runner):
    """It is not host work: the device computes it, which costs nothing."""
    out = np.zeros(4, np.float64)
    runner(device_member_above_loop, out)
    np.testing.assert_allclose(out, np.arange(4) + 7.0)
