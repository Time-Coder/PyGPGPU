"""What a pass boundary copies: nothing unless a host region needs the bytes.

Between two passes the device already holds the newest content, so the bytes only have to
come home when a host region (or the caller) actually reads them. These tests pin both
halves of that policy for both array flavours the package accepts:

* a boundary whose region names no array copies **nothing** home -- the arrays stay on the
  device and come home once after the last pass;
* a boundary whose region reads an array copies **that** array home before the region runs,
  because the region is python and must see the previous pass's output.

They read the copy trace the backends print under `set_print_info(True)`, which is the only
place a copy is observable from the host: the array's own contents look the same either way.
"""
from __future__ import annotations

import numpy as np
import pytest

import pygpgpu.numpy as gnp
from pygpgpu import kernel, set_print_info


@kernel
def two_passes_untouched(a: np.ndarray, b: np.ndarray):
    """No statement between the loops names either array."""
    for i in range(a.shape[0]):
        a[i] = a[i] * 2.0
    for i in range(a.shape[0]):
        b[i] = a[i] + 1.0


@kernel
def two_passes_region_reads_a(a: np.ndarray, b: np.ndarray):
    """The region between the loops reads the first pass's output."""
    for i in range(a.shape[0]):
        a[i] = a[i] * 2.0
    shift = int(a[0])
    for i in range(a.shape[0]):
        b[i] = a[i] + float(shift)


@kernel
def two_passes_region_reads_through_alias(a: np.ndarray, b: np.ndarray):
    """The region reaches `a` only through a local alias, never by its own name."""
    for i in range(a.shape[0]):
        a[i] = a[i] * 2.0
    view = a
    shift = int(view[0])
    for i in range(a.shape[0]):
        b[i] = a[i] + float(shift)


@kernel
def two_passes_region_builds_alias_chain(a: np.ndarray, b: np.ndarray):
    """A two-step alias chain: the region mentions `a` only in the first binding."""
    for i in range(a.shape[0]):
        a[i] = a[i] * 2.0
    outer = a
    inner = outer
    shift = int(inner[0])
    for i in range(a.shape[0]):
        b[i] = a[i] + float(shift)


def _copy_trace(capsys, text: str) -> list[str]:
    """The copy and launch lines the backend printed, in order.

    The launch lines matter as much as the copy lines: whether a copy happened *before* the
    launch that consumes it is what separates "kept on the device" from "round-tripped".
    """
    return [line.strip() for line in text.splitlines()
            if "copy data to " in line or "launch" in line]


def _launch_positions(lines: list[str]) -> list[int]:
    """The indexes of the launch lines in a trace."""
    return [index for index, line in enumerate(lines) if line.startswith("launch")]


@pytest.fixture
def copy_trace(capsys):
    """Run one launch and report the copy lines in the order the backend printed them."""
    set_print_info(True)
    try:
        yield lambda: _copy_trace(capsys, capsys.readouterr().out)
    finally:
        set_print_info(False)


@pytest.mark.parametrize("flavour", ["plain", "gnp"])
def test_an_untouched_boundary_copies_nothing_home(gpu_backend, copy_trace, flavour):
    """Two passes, no host read in between: the bytes come home once, after the last pass.

    The count alone is not enough: a launch that reads an array back between the passes and
    uploads it again still ends with one home copy, so the *position* of the copies is what
    pins the policy. Everything that goes home must come after the second launch, and
    nothing may be uploaded after the first one -- an upload there means the bytes came home
    and went back out for a boundary that never needed them.
    """
    if flavour == "gnp":
        a = gnp.arange(4, dtype=np.float32)
        b = gnp.zeros(4, dtype=np.float32)
    else:
        a = np.arange(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
    two_passes_untouched[gpu_backend](a, b)
    lines = copy_trace()
    np.testing.assert_array_equal(a, np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(b, np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32))
    if gpu_backend == "python":
        return                      # the interpreter copies nothing; there is no device
    assert lines.count("copy data to host for argument 'a'") == 1, lines
    assert lines.count("copy data to host for argument 'b'") == 1, lines
    # Every home copy sits after the last launch, and no upload follows the first one.
    launches = _launch_positions(lines)
    assert len(launches) == 2, lines
    first_home = min(index for index, line in enumerate(lines)
                     if line.startswith("copy data to host"))
    assert first_home > launches[-1], lines


@pytest.mark.parametrize("flavour", ["plain", "gnp"])
def test_a_boundary_that_reads_an_array_copies_it_home_first(gpu_backend, copy_trace,
                                                            flavour):
    """The region is python, so the array it reads must be current before it runs.

    `a` is read by the region and must come home before the second launch; `b` is only
    written, so it still comes home once at the very end. The values prove the ordering was
    right: the region saw the first pass's output (shift == 0) rather than the caller's
    input (shift == 1).
    """
    if flavour == "gnp":
        a = gnp.arange(4, dtype=np.float32)
        b = gnp.zeros(4, dtype=np.float32)
    else:
        a = np.arange(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
    two_passes_region_reads_a[gpu_backend](a, b)
    np.testing.assert_array_equal(a, np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(b, np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32))
    if gpu_backend == "python":
        return
    lines = copy_trace()
    launches = _launch_positions(lines)
    home_a = [index for index, line in enumerate(lines)
              if line == "copy data to host for argument 'a'"]
    assert home_a, lines
    assert len(launches) == 2, lines
    # `a` is read by the region, so its read-back must precede the second launch; `b` is
    # only written, so its read-back stays at the end.
    assert home_a[0] < launches[-1], lines
    home_b = [index for index, line in enumerate(lines)
              if line == "copy data to host for argument 'b'"]
    assert home_b and home_b[-1] > launches[-1], lines


def test_the_interpreter_needs_no_copies(gpu_backend, copy_trace):
    """The python backend has no device, so nothing is copied in either direction."""
    if gpu_backend != "python":
        pytest.skip("this pins the interpreter's own behaviour")
    a = np.arange(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    two_passes_untouched[gpu_backend](a, b)
    np.testing.assert_array_equal(b, np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32))
    assert copy_trace() == []


@pytest.mark.parametrize("kernel_under_test", [
    two_passes_region_reads_a,
    two_passes_region_reads_through_alias,
    two_passes_region_builds_alias_chain,
], ids=["plain-name", "alias", "alias-chain"])
@pytest.mark.parametrize("flavour", ["plain", "gnp"])
def test_a_region_that_reaches_the_array_indirectly_still_sees_it(
        gpu_backend, copy_trace, kernel_under_test, flavour):
    """An alias must not hide the array from the read-back decision.

    The touch analysis asks whether a region *mentions* the array: `view = a` names `a` on
    its right-hand side, so the region is known to need it even though every later read goes
    through `view`. What the values prove is the consequence -- the region sees the first
    pass's output (shift 0), not the caller's input (shift 1) -- which is the thing that
    breaks the moment the analysis misses the array and the bytes stay on the device.
    """
    if flavour == "gnp":
        a = gnp.arange(4, dtype=np.float32)
        b = gnp.zeros(4, dtype=np.float32)
    else:
        a = np.arange(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
    kernel_under_test[gpu_backend](a, b)
    np.testing.assert_array_equal(np.asarray(a),
                                  np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(b),
                                  np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32))
    if gpu_backend == "python":
        return
    lines = copy_trace()
    home_a = [index for index, line in enumerate(lines)
              if line == "copy data to host for argument 'a'"]
    assert home_a, lines
    launches = _launch_positions(lines)
    assert launches, lines
    # The read-back has to happen before the launch that consumes it.
    assert home_a[0] < launches[-1], lines


def test_a_strided_array_is_written_back_to_its_own_buffer(gpu_backend):
    """A non-contiguous view must keep the immediate read-back, and land in the right place.

    The deferred path writes the device bytes back through the view that was bound, and a
    strided array's device copy is a *compacted* staging buffer: writing it back that way
    left the caller's view unmodified and scrambled the buffer behind it. Such an array is
    therefore excluded from the deferred set, which is what keeps this case correct.
    """
    backing = np.zeros(8, dtype=np.float32)
    backing[::2] = np.arange(4, dtype=np.float32)
    view = backing[::2]
    b = np.zeros(4, dtype=np.float32)
    two_passes_untouched[gpu_backend](view, b)
    np.testing.assert_array_equal(np.asarray(view),
                                  np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(backing,
                                  np.array([0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0],
                                           dtype=np.float32))
    np.testing.assert_array_equal(b, np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32))
