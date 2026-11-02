"""Every work-size bracket states a CUDA launch: ``(blocks_per_grid, threads_per_block)``.

One spelling on every backend, because it is what a CUDA or HIP reader already writes. An
OpenCL reader states a group width through `local=` in its own API rather than through a
bracket, so nothing is lost by agreeing on one form here.

Three entries reach the same bracket, and all three agree:

* ``@kernel``: ``k[(4, 256)]`` is 4 blocks of 256 threads -- 1024 work items;
* the low-level `oop` ``Kernel``: ``kernel[(4, 256)]`` is the same launch;
* a kernel imported from ``.cl`` / ``.cu`` / ``.hip``: the same ``Kernel`` object.

OpenCL's enqueue call speaks a *different* pair (total item count, one group's width), so
``utils.blocks_to_global`` converts once, at the bracket. Two things follow: the total
always divides by the group width (the quotient *is* the block count), so there is no
divisibility rule left to enforce; and a CUDA/HIP launch derives ``grid = global // local``,
which is ``blocks`` by construction, so the three backends launch the same grid.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel, get_global_id, get_group_id, get_local_id
from pygpgpu.utils import blocks_to_global


# ---------------------------------------------------------------------------
# The conversion itself
# ---------------------------------------------------------------------------


def test_a_bracket_is_blocks_and_threads():
    """`(4, 256)` is 1024 work items split into 256-wide groups -- i.e. 4 blocks."""
    total, threads = blocks_to_global((4,), (256,))
    assert total == (1024,)
    assert threads == (256,)


def test_the_conversion_needs_no_divisibility():
    """Any block count multiplied by a thread count divides exactly: the quotient is blocks."""
    for blocks, threads in [((3,), (7,)), ((3912,), (256,)), ((5, 7), (3, 11))]:
        total, group = blocks_to_global(blocks, threads)
        assert tuple(t // g for t, g in zip(total, group)) == tuple(blocks)


def test_the_two_axes_have_to_be_the_same_rank():
    """Two axes cannot say how many threads a three-axis grid holds."""
    import pytest

    with pytest.raises(ValueError) as caught:
        blocks_to_global((2, 3), (8,))
    assert "same" in str(caught.value)


# ---------------------------------------------------------------------------
# @kernel's bracket
# ---------------------------------------------------------------------------


@kernel(domain="out")
def writes_its_global_id(out: np.ndarray):
    """One element per work item, so a wrong grid shows up as an untouched tail."""
    i = get_global_id(0)
    out[i] = i


def test_a_kernel_bracket_launches_blocks_of_threads(gpu_backend):
    """`(4, 16)` covers 64 items: 4 blocks, 16 threads each."""
    out = np.full(64, -1.0, dtype=np.float32)
    writes_its_global_id[gpu_backend][(4, 16)](out)
    np.testing.assert_array_equal(out, np.arange(64, dtype=np.float32))


def test_a_kernel_bracket_reports_the_group_it_names(gpu_backend):
    """`threadIdx.x` runs 0..threads-1 and `blockIdx.x` runs 0..blocks-1."""

    @kernel(domain="out")
    def ids(out: np.ndarray):
        i = get_global_id(0)
        out[i] = get_group_id(0) * 100 + get_local_id(0)

    out = np.zeros(32, dtype=np.int32)
    writes = ids
    writes[gpu_backend][(4, 8)](out)
    expected = np.array([blk * 100 + lane for blk in range(4) for lane in range(8)],
                        dtype=np.int32)
    np.testing.assert_array_equal(out[:32], expected)


def test_one_bracket_number_is_a_bare_item_count():
    """`k[(n,)]` / `k[n]` stay a plain count with an automatic group size."""
    from pygpgpu.kernel.decorator import _parse_work_item

    assert _parse_work_item((64,)) == ((64,), None)
    assert _parse_work_item(64) == ((64,), None)


def test_a_bracket_states_the_launch_in_cuda_terms():
    """`(blocks, threads)` -> the total-work-item pair the backends use internally."""
    from pygpgpu.kernel.decorator import _parse_work_item

    assert _parse_work_item((4, 256)) == ((1024,), (256,))


# ---------------------------------------------------------------------------
# The low-level oop Kernel, on every backend that has one
# ---------------------------------------------------------------------------


_SOURCE = """
__kernel void marks_its_own_group(__global int* out)
{
    int i = get_global_id(0);
    out[i] = get_group_id(0) * 100 + get_local_id(0);
}
"""


def _opencl_kernel(tmp_path):
    """Compile an OpenCL program from a file and hand back its Kernel."""
    from pygpgpu.opencl.oop.platforms import Platforms

    source_file = tmp_path / "bracket.cl"
    source_file.write_text(_SOURCE, encoding="utf-8")
    context = Platforms[0].default_context
    program = context.compile(str(source_file))
    return program.marks_its_own_group


def test_the_opencl_oop_kernel_takes_blocks_and_threads(tmp_path, gpu_backend):
    """The low-level bracket means the same thing the decorator's does."""
    if gpu_backend != "opencl":
        import pytest
        pytest.skip("the oop Kernel under test is OpenCL's")
    kernel_object = _opencl_kernel(tmp_path)
    out = np.zeros(32, dtype=np.int32)
    kernel_object[(4, 8)](out)
    expected = np.array([blk * 100 + lane for blk in range(4) for lane in range(8)],
                        dtype=np.int32)
    np.testing.assert_array_equal(out, expected)


def test_the_opencl_oop_bracket_rejects_two_unrelated_axes(tmp_path, gpu_backend):
    """A block count and a thread count have to name the same number of axes."""
    import pytest

    if gpu_backend != "opencl":
        pytest.skip("the oop Kernel under test is OpenCL's")
    kernel_object = _opencl_kernel(tmp_path)
    with pytest.raises(ValueError) as caught:
        kernel_object[((2, 3), 8)]
    assert "same dimension" in str(caught.value) or "same" in str(caught.value)
