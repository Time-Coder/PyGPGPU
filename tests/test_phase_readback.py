"""A pass boundary without a host read leaves the data on the device."""
import numpy as np
from pygpgpu import kernel


@kernel
def two_passes_no_gap_read(a: np.ndarray, b: np.ndarray):
    """No statement between the loops names either array."""
    for i in range(a.shape[0]):
        a[i] = a[i] * 2.0
    for i in range(a.shape[0]):
        b[i] = a[i] + 1.0


@kernel
def two_passes_gap_reads(a: np.ndarray, b: np.ndarray):
    """The region between the loops reads the first pass's output."""
    for i in range(a.shape[0]):
        a[i] = a[i] * 2.0
    shift = int(a[0])
    for i in range(a.shape[0]):
        b[i] = a[i] + float(shift)


def test_a_pass_boundary_keeps_the_arrays_on_the_device(gpu_backend, capsys):
    """Neither array is copied home between the passes.

    Between two passes the device already holds the newest bytes, so a read-back followed
    by an upload copies twice for nothing unless a host region touches the array. The
    arrays that a region never names travel through the deferred-read-back view; their
    bytes come home once, after the last pass.
    """
    a = np.arange(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    two_passes_no_gap_read[gpu_backend](a, b)
    np.testing.assert_array_equal(a, np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(b, np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32))
    if gpu_backend in ("cuda", "hip", "opencl"):
        printed = capsys.readouterr().out
        # Two uploads per array (one per launch at most) and one home per array.
        assert printed.count("copy data to host for argument 'a'") <= 1, printed
        assert printed.count("copy data to host for argument 'b'") == 1, printed


def test_a_gap_read_still_sees_the_previous_pass(gpu_backend):
    """A region that reads the array forces its bytes home, and answers python's value."""
    a = np.arange(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    two_passes_gap_reads[gpu_backend](a, b)
    np.testing.assert_array_equal(a, np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(b, np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32))
