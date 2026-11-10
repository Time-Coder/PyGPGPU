"""Work-group storage a gid-style body indexes directly: `local_array` / `shared_array`.

Two pairs of builtins build one device declaration each (`__shared__ T name[N]` on CUDA/HIP,
one `__local`/`threadgroup` parameter on OpenCL and Metal), and they differ in what the *name*
in the host model stands for:

* `local_array_per_group` / `shared_array_per_block` build a value that stands for every
  group's slice, so a body indexes it by its group first (`buf[blockIdx][j]`). Its sibling
  test file is `test_kernel_local_memory.py`.
* `local_array` / `shared_array` build **this block's own** array, indexed directly
  (`buf[j]`). That is only meaningful where the body is one work item -- the gid style -- so
  a magic-loop body is refused, which is the other half of what these tests pin.

The two pairs are separate functions, not one function under two names: an editor shows a
call to the one that was written, and the translator's error messages name the same spelling.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import (
    atomic_add,
    barrier,
    blockIdx,
    get_global_id,
    get_local_id,
    kernel,
    local_array,
    local_array_per_group,
    shared_array,
    shared_array_per_block,
)

LANES = 8


@kernel(domain="pixels")
def direct_histogram(pixels: np.ndarray, hist: np.ndarray):
    """CUDA's shared-memory histogram, with no group subscript at all."""
    lane = get_local_id(0)
    bins = local_array(LANES, np.uint32)
    bins[lane] = 0
    barrier()
    atomic_add(bins, int(pixels[get_global_id(0)]), 1)
    barrier()
    if lane < LANES:
        atomic_add(hist, lane, bins[lane])


@kernel(domain="pixels")
def direct_histogram_plus_equals(pixels: np.ndarray, hist: np.ndarray):
    """The same kernel with `+=`, which is atomic on work-group storage for the same reason."""
    lane = get_local_id(0)
    bins = local_array(LANES, np.uint32)
    bins[lane] = 0
    barrier()
    bins[int(pixels[get_global_id(0)])] += 1
    barrier()
    if lane < LANES:
        hist[lane] += bins[lane]


@kernel(domain="x")
def block_scratch(x: np.ndarray, out: np.ndarray):
    """One slot per lane, written and then read back by a later work item of the group."""
    lane = get_local_id(0)
    slots = shared_array(LANES, np.int32)
    slots[lane] = int(x[get_global_id(0)]) * 2
    barrier()
    out[get_global_id(0)] = slots[lane]


@kernel(domain="out")
def grid_slots(out: np.ndarray):
    """A two-dimensional per-block buffer: `buf[i, j]` flattens like any array's."""
    lane = get_local_id(0)
    grid = local_array((2, 4), np.int32)
    grid[lane // 4, lane % 4] = lane
    barrier()
    out[get_global_id(0)] = grid[lane // 4, lane % 4]


@kernel(domain="out")
def per_group_grid_slots(out: np.ndarray):
    """The per-group twin of `grid_slots`: the same buffer, indexed by its group first."""
    lane = get_local_id(0)
    grid = local_array_per_group((2, 4), np.int32)
    grid[blockIdx][lane // 4, lane % 4] = lane
    barrier()
    out[get_global_id(0)] = grid[blockIdx][lane // 4, lane % 4]


def test_a_work_group_histogram_matches_numpy(gpu_backend: str) -> None:
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, LANES, size=16 * 1024, dtype=np.uint8)
    hist = np.zeros(LANES, dtype=np.uint32)
    pixel_blocks = -(-pixels.size // LANES)
    direct_histogram[gpu_backend][(pixel_blocks, LANES)](pixels, hist)
    np.testing.assert_array_equal(
        hist.astype(np.int64), np.histogram(pixels, bins=LANES, range=(0, LANES))[0])


def test_the_augmented_spelling_is_atomic_on_block_storage(gpu_backend: str) -> None:
    rng = np.random.default_rng(17)
    pixels = rng.integers(0, LANES, size=8 * 1024, dtype=np.uint8)
    hist = np.zeros(LANES, dtype=np.uint32)
    pixel_blocks = -(-pixels.size // LANES)
    direct_histogram_plus_equals[gpu_backend][(pixel_blocks, LANES)](pixels, hist)
    np.testing.assert_array_equal(
        hist.astype(np.int64), np.histogram(pixels, bins=LANES, range=(0, LANES))[0])


def test_a_block_buffer_is_usable_as_per_lane_storage(gpu_backend: str) -> None:
    values = np.arange(LANES, dtype=np.int32)
    out = np.full(LANES, -1, dtype=np.int32)
    block_scratch[gpu_backend][(1, LANES)](values, out)
    np.testing.assert_array_equal(out, values * 2)


def test_a_two_dimensional_block_buffer_flattens_like_an_array(gpu_backend: str) -> None:
    out = np.zeros(LANES, dtype=np.int32)
    grid_slots[gpu_backend][(1, LANES)](out)
    np.testing.assert_array_equal(out, np.arange(LANES, dtype=np.int32))


def test_the_two_spellings_of_a_work_group_buffer_agree(gpu_backend: str) -> None:
    """One declaration, two index spellings: `buf[i, j]` and `buf[blockIdx][i, j]`."""
    direct = np.zeros(LANES, dtype=np.int32)
    grouped = np.zeros(LANES, dtype=np.int32)
    grid_slots[gpu_backend][(1, LANES)](direct)
    per_group_grid_slots[gpu_backend][(1, LANES)](grouped)
    np.testing.assert_array_equal(direct, grouped)


def test_the_interpreter_keeps_one_array_per_block() -> None:
    """A serial run still shares the array between the work items of one block."""

    @kernel(domain="x")
    def tag(x: np.ndarray, out: np.ndarray):
        lane = get_local_id(0)
        buf = local_array(4, np.int32)
        buf[lane % 4] = int(x[get_global_id(0)]) // 4 * 100 + lane % 4
        barrier()
        out[get_global_id(0)] = buf[lane % 4]

    out = np.zeros(8, dtype=np.int32)
    tag["python"][(2, 4)](np.arange(8, dtype=np.int32), out)
    np.testing.assert_array_equal(
        out, np.array([0, 1, 2, 3, 100, 101, 102, 103], dtype=np.int32))


def test_two_declarations_in_one_body_are_two_arrays() -> None:
    """The storage is keyed by the statement that asked for it, not by the block alone."""

    @kernel(domain="out")
    def two(out: np.ndarray):
        i = get_global_id(0)
        first = local_array(4, np.int32)
        second = local_array(4, np.int32)
        first[i] = 1
        second[i] = 2
        barrier()
        out[i] = first[i] * 10 + second[i]

    out = np.zeros(4, dtype=np.int32)
    two["python"][(1, 4)](out)
    np.testing.assert_array_equal(out, np.array([12, 12, 12, 12], dtype=np.int32))


def test_a_magic_loop_body_is_refused(gpu_backend: str) -> None:
    """A magic loop stands for the whole launch, so a bare block array has no block."""

    @kernel
    def wrong_style(out: np.ndarray):
        buf = local_array(4, np.float32)
        for i in range(out.shape[0]):
            buf[i] = 1.0
            barrier()
            out[i] = buf[i]

    with pytest.raises(SyntaxError) as caught:
        wrong_style[gpu_backend](np.zeros(4, dtype=np.float32))
    message = str(caught.value)
    assert "gid style" in message, message
    assert "local_array_per_group" in message, message


def test_a_body_with_no_work_item_is_refused(gpu_backend: str) -> None:
    @kernel
    def no_work_item(out: np.ndarray):
        buf = local_array(4, np.float32)
        buf[0] = 1.0
        out[0] = buf[0]

    with pytest.raises(SyntaxError) as caught:
        no_work_item[gpu_backend](np.zeros(4, dtype=np.float32))
    assert "needs a work item" in str(caught.value)


def test_a_block_array_shape_must_be_constant(gpu_backend: str) -> None:
    factor = 2

    @kernel(domain="out")
    def computed(out: np.ndarray, width: int):
        size = width * factor
        buf = local_array(size, np.float32)
        buf[get_local_id(0)] = 1.0
        barrier()
        out[get_global_id(0)] = buf[get_local_id(0)]

    with pytest.raises(SyntaxError) as caught:
        computed[gpu_backend][(1, 4)](np.zeros(4, dtype=np.float32), 4)
    assert "constant positive integer" in str(caught.value)


def test_a_non_integer_index_is_refused(gpu_backend: str) -> None:
    @kernel(domain="out")
    def float_index(out: np.ndarray):
        buf = local_array(4, np.float32)
        buf[0.5] = 1.0
        out[0] = buf[0.0]

    with pytest.raises(SyntaxError) as caught:
        float_index[gpu_backend][(1, 4)](np.zeros(4, dtype=np.float32))
    assert "integer" in str(caught.value)


def test_the_two_pairs_are_four_distinct_objects() -> None:
    """No aliases: each spelling is its own function, so an editor highlights the call made."""
    assert local_array is not local_array_per_group
    assert shared_array is not shared_array_per_block
    assert shared_array is not local_array
    assert shared_array_per_block is not local_array_per_group


def test_the_rename_kept_one_signature_per_name() -> None:
    """Both names of a pair take `np.zeros`' arguments, in `np.zeros`' order."""
    import inspect

    for function in (local_array, shared_array, local_array_per_group,
                     shared_array_per_block):
        parameters = list(inspect.signature(function).parameters)
        assert parameters == ["shape", "dtype"], (function.__name__, parameters)
