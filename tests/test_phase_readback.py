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


def _copy_trace(capsys, text: str) -> list[str]:
    """The copy lines the backend printed, in order."""
    return [line.strip() for line in text.splitlines() if "copy data to " in line]


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
    # Every home copy sits at the end, and no upload follows the first launch.
    first_home = min(index for index, line in enumerate(lines) if line.startswith("copy data to host"))
    assert first_home >= len(lines) - 2, lines
    assert not any(line.startswith("copy data to host") for line in lines[:first_home]), lines


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
    assert "copy data to host for argument 'a'" in lines, lines
    first_home_a = lines.index("copy data to host for argument 'a'")
    # `a`'s home copy precedes the last home copy of `b`, which is the end of the launch.
    assert first_home_a < len(lines) - 1, lines


def test_the_interpreter_needs_no_copies(gpu_backend, copy_trace):
    """The python backend has no device, so nothing is copied in either direction."""
    if gpu_backend != "python":
        pytest.skip("this pins the interpreter's own behaviour")
    a = np.arange(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    two_passes_untouched[gpu_backend](a, b)
    np.testing.assert_array_equal(b, np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32))
    assert copy_trace() == []
