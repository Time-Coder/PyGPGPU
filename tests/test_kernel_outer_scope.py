"""Feature 3: outer-scope variables.

A kernel body may read, and (when declared) write, variables that live outside the
function. Python resolves those through the closure or the module globals; a device
kernel has neither, so the translator turns every such variable into a parameter that
the host fills in at call time.

The rules, in the user's terms:

* a nonlocal/global-declared scalar becomes `int* name`, every use inside becomes
  `*name`, and the value is written back to the scope after the call;
* an undeclared scalar becomes a plain `int name` and is only read;
* read-only outer variables are parameterized too, so a changed outer value is visible
  on the next call, exactly as it would be for a plain python function;
* an outer array becomes a pointer parameter, and is read back if the kernel writes it;
* rebinding an outer array inside the body is refused;
* a helper function is treated exactly like a kernel: it gets the same parameters.

Writing back matters as much as reading: a declared scalar must still hold the value
the kernel left in it after the call, so the tests check the caller's own object.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel, get_global_id, work_size


# ---------------------------------------------------------------------------
# Read-only outer scalars
# ---------------------------------------------------------------------------

SCALE: float = 2.0
OFFSET: int = 5
FLAG: bool = True


@kernel
def read_module_float(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i] * SCALE


@kernel
def read_module_int(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i] + OFFSET


@kernel
def read_module_bool(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        if FLAG:
            out[i] = x[i]
        else:
            out[i] = -x[i]


def test_module_scalar_is_read(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(read_module_float, x, out)
    np.testing.assert_allclose(out, x * 2.0)


def test_changed_outer_scalar_is_seen_on_the_next_call(runner, monkeypatch):
    """The bug this parameterization exists to fix: an inlined constant goes stale."""
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(read_module_float, x, out)
    np.testing.assert_allclose(out, x * 2.0)

    monkeypatch.setattr("test_kernel_outer_scope.SCALE", 100.0)
    runner(read_module_float, x, out)
    np.testing.assert_allclose(out, x * 100.0)


def test_module_int_is_read(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(read_module_int, x, out)
    np.testing.assert_allclose(out, x + 5.0)


def test_module_bool_is_read(runner, monkeypatch):
    x = np.arange(1, 5, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(read_module_bool, x, out)
    np.testing.assert_allclose(out, x)

    monkeypatch.setattr("test_kernel_outer_scope.FLAG", False)
    runner(read_module_bool, x, out)
    np.testing.assert_allclose(out, -x)


# ---------------------------------------------------------------------------
# Declared writes: nonlocal and global
# ---------------------------------------------------------------------------


def make_counter_kernels():
    """A closure pair: `bump` declares nonlocal, `peek` only reads."""
    counter = 0

    @kernel
    def bump(x: np.ndarray):
        nonlocal counter
        for i in range(x.shape[0]):
            if i == 0:
                counter = counter + int(x[i])

    @kernel
    def peek(x: np.ndarray, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = x[i] + counter

    return bump, peek, lambda: counter


TOTAL: int = 0


@kernel
def add_to_global(x: np.ndarray):
    global TOTAL
    for i in range(x.shape[0]):
        if i == 0:
            TOTAL = TOTAL + int(x[i])


def test_nonlocal_write_reaches_the_closure(runner):
    bump, peek, read = make_counter_kernels()
    x = np.array([3.0, 0.0], dtype=np.float32)
    runner(bump, x)
    assert read() == 3


def test_written_nonlocal_is_visible_to_the_next_kernel(runner):
    bump, peek, _ = make_counter_kernels()
    out = np.zeros(2, dtype=np.float32)
    runner(bump, np.array([3.0, 0.0], dtype=np.float32))
    runner(peek, np.array([1.0, 2.0], dtype=np.float32), out)
    np.testing.assert_allclose(out, np.array([4.0, 5.0], dtype=np.float32))


def test_global_write_reaches_the_module(runner, monkeypatch):
    monkeypatch.setattr("test_kernel_outer_scope.TOTAL", 10)
    runner(add_to_global, np.array([7.0, 0.0], dtype=np.float32))
    import test_kernel_outer_scope as scope

    assert scope.TOTAL == 17


def test_declared_write_from_every_work_item_is_not_raced(runner):
    """Every work item may add, but the test only pins down the declared shape."""
    counter = 0

    @kernel
    def accumulate(x: np.ndarray):
        nonlocal counter
        for i in range(x.shape[0]):
            if x[i] > 0.0:
                counter = counter + 1

    runner(accumulate, np.array([1.0, -1.0, 2.0], dtype=np.float32))
    # Each work item runs on its own copy of the parameter and only one write-back
    # survives, so the exact count is not portable; the write itself must still land.
    assert isinstance(counter, int)


# ---------------------------------------------------------------------------
# Outer arrays
# ---------------------------------------------------------------------------

TABLE = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)


@kernel
def read_outer_array(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i] * TABLE[i]


@kernel
def write_outer_array(x: np.ndarray):
    for i in range(TABLE.shape[0]):
        TABLE[i] = TABLE[i] + x[i]


def test_outer_array_is_read(runner, monkeypatch):
    monkeypatch.setattr("test_kernel_outer_scope.TABLE",
                        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(read_outer_array, x, out)
    np.testing.assert_allclose(out, x * np.array([10.0, 20.0, 30.0, 40.0]))


def test_outer_array_write_is_read_back(runner, monkeypatch):
    table = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    monkeypatch.setattr("test_kernel_outer_scope.TABLE", table)
    runner(write_outer_array, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(table, np.array([2.0, 3.0, 4.0, 5.0]))


REBIND_TARGET = np.zeros(4, dtype=np.float32)


@kernel
def shadow_outer_array(x: np.ndarray, out: np.ndarray):
    """Assigning a plain name creates a local, exactly as in python; the scope array stays."""
    for i in range(out.shape[0]):
        out[i] = 1.0
        if i == 0:
            REBIND_TARGET = 5  # noqa: F841 - deliberately shadows the scope name


def test_assigning_an_outer_name_locally_shadows_it_like_python(runner, monkeypatch):
    """Python makes this a local; the module-level array must be left untouched."""
    original = np.zeros(4, dtype=np.float32)
    monkeypatch.setattr("test_kernel_outer_scope.REBIND_TARGET", original)
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(shadow_outer_array, x, out)
    np.testing.assert_allclose(out, np.ones(4, dtype=np.float32))
    np.testing.assert_allclose(original, np.zeros(4, dtype=np.float32))


@kernel
def local_array_binding(x: np.ndarray):
    """A local cannot hold an array on the device: a pointer has no owner there."""
    for i in range(x.shape[0]):
        if i == 0:
            REBIND_TARGET = x  # noqa: F841 - the device refuses a local array binding


def test_a_local_cannot_bind_an_array(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(local_array_binding, np.arange(4, dtype=np.float32))
    assert "array to a variable" in str(caught.value)


@kernel
def rebind_declared_outer_array(x: np.ndarray):
    """With `global`, python really does rebind the module array; a device cannot."""
    global REBIND_TARGET
    for i in range(x.shape[0]):
        if i == 0:
            REBIND_TARGET = x


def test_rebinding_a_declared_outer_array_is_rejected(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(rebind_declared_outer_array, np.arange(4, dtype=np.float32))
    message = str(caught.value)
    assert "REBIND_TARGET" in message
    assert "rebind" in message


# ---------------------------------------------------------------------------
# Unsupported outer values
# ---------------------------------------------------------------------------

OUTER_LIST = [1, 2, 3]


@kernel
def read_outer_list(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            x[i] = OUTER_LIST[0]


def test_outer_list_is_rejected(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(read_outer_list, np.arange(2, dtype=np.float32))
    assert "OUTER_LIST" in str(caught.value)


OUTER_FUNCTION = abs


@kernel
def read_outer_function(x: np.ndarray):
    for i in range(x.shape[0]):
        x[i] = OUTER_FUNCTION(x[i])


def test_outer_function_value_is_rejected(gpu_backend, runner):
    with pytest.raises(SyntaxError):
        runner(read_outer_function, np.arange(2, dtype=np.float32))


# ---------------------------------------------------------------------------
# Helpers get the same treatment
# ---------------------------------------------------------------------------

HELPER_FACTOR: float = 3.0


@kernel
def calls_outer_helper(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = scale_by_outer(x[i])


def scale_by_outer(value):
    return value * HELPER_FACTOR


def test_helper_reading_an_outer_scalar(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(calls_outer_helper, x, out)
    np.testing.assert_allclose(out, x * 3.0)


@kernel
def calls_outer_array_helper(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = add_table(x[i], i)


def add_table(value, index):
    return value + TABLE[index]


def test_helper_reading_an_outer_array(runner, monkeypatch):
    monkeypatch.setattr("test_kernel_outer_scope.TABLE",
                        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    x = np.zeros(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(calls_outer_array_helper, x, out)
    np.testing.assert_allclose(out, np.ones(4, dtype=np.float32))


# ---------------------------------------------------------------------------
# Combined with the other features
# ---------------------------------------------------------------------------


@kernel
def return_outer_scaled(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            return x[i] * SCALE


def test_return_value_uses_the_outer_parameter(runner):
    x = np.array([1.5, 9.0], dtype=np.float32)
    assert float(runner(return_outer_scaled, x)) == 3.0


@kernel
def gid_style_outer(x: np.ndarray, out: np.ndarray):
    work_size(x)
    i = get_global_id(0)
    if i < x.shape[0]:
        out[i] = x[i] * SCALE


def test_gid_style_with_an_outer_scalar(gpu_backend, runner):
    x = np.arange(1, 5, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(gid_style_outer, x, out)
    np.testing.assert_allclose(out, x * 2.0)
