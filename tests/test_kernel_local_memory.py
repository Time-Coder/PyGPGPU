"""Work-group shared memory: `zeros_per_group(...)`, `barrier()`, and atomics on those buffers.

This is the `__shared__` histogram every CUDA tutorial shows, written once and run on every
device backend:

.. code-block:: cuda

    __shared__ unsigned int bins[256];
    bins[threadIdx.x] = 0;
    __syncthreads();
    atomicAdd(&bins[pixel], 1);
    __syncthreads();
    if (threadIdx.x < 256) atomicAdd(&global[tid], bins[tid]);

A `zeros_per_group(...)` value stands for **every group's slice**, so it is indexed by the group first
(`bins[blockIdx][lane]`) and only then by the slot. On a device that first subscript folds
away -- the hardware already gives each group private storage -- but the spelling keeps the
model honest, and it is what makes the same code meaningful on the host.

Three things had to line up for the histogram to work: the CUDA spelling of the shared array
(`uint` is not defined by NVRTC), the `__local` address space in the OpenCL atomic cast
(casting `__local` to `__global` does not compile), and a barrier that fences local memory.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import (
    atomic_add,
    barrier,
    blockIdx,
    get_global_id,
    get_group_id,
    get_local_id,
    get_local_size,
    kernel,
    zeros_per_group,
    work_size,
)

BINS = 256


@kernel
def shared_histogram(pixels: np.ndarray, hist: np.ndarray):
    """One work item per pixel; counts accumulate in shared memory, then merge once."""
    work_size(pixels)              # bounds guard against the real pixel count
    lane = get_local_id(0)         # threadIdx.x
    bins = zeros_per_group(np.uint32, 256)   # __shared__ uint bins[256], per group

    bins[blockIdx][lane] = 0       # zero this group's bins (one lane per bin)
    barrier()                      # __syncthreads()

    atomic_add(bins[blockIdx], int(pixels[get_global_id(0)]), 1)
    barrier()                      # __syncthreads()

    if lane < 256:
        atomic_add(hist, lane, bins[blockIdx][lane])   # merge into the global histogram


@kernel
def shared_histogram_plus_equals(pixels: np.ndarray, hist: np.ndarray):
    """The same kernel with `+=`, which is atomic on shared memory for the same reason."""
    work_size(pixels)
    lane = get_local_id(0)
    bins = zeros_per_group(np.uint32, 256)
    bins[blockIdx][lane] = 0
    barrier()
    bins[blockIdx][int(pixels[get_global_id(0)])] += 1
    barrier()
    if lane < 256:
        hist[lane] += bins[blockIdx][lane]


@kernel
def shared_sums(x: np.ndarray, hist: np.ndarray):
    """A shared buffer used as a scratch pad, without atomics: each lane owns one slot."""
    work_size(x)
    lane = get_local_id(0)
    slots = zeros_per_group(np.int32, 256)
    slots[blockIdx][lane] = int(x[get_global_id(0)]) * 2
    barrier()
    if lane < 256:
        hist[lane] = slots[blockIdx][lane]


def _launch_shared(function, backend, pixels, hist):
    """A group of 256 lanes is required: one lane per bin, as in the CUDA version."""
    total = -(-pixels.size // BINS) * BINS
    function[backend][(total, BINS)](pixels, hist)


def _expected(pixels: np.ndarray) -> np.ndarray:
    return np.histogram(pixels, bins=BINS, range=(0, BINS))[0]


def test_a_shared_memory_histogram_matches_numpy(gpu_backend):
    rng = np.random.default_rng(11)
    pixels = rng.integers(0, BINS, size=64 * 1024, dtype=np.uint8)
    hist = np.zeros(BINS, dtype=np.uint32)
    _launch_shared(shared_histogram, gpu_backend, pixels, hist)
    np.testing.assert_array_equal(hist.astype(np.int64), _expected(pixels))


def test_the_augmented_spelling_is_atomic_on_shared_memory(gpu_backend):
    """`bins[value] += 1` writes to `__local` memory, so the atomic cast must say so."""
    rng = np.random.default_rng(13)
    pixels = rng.integers(0, BINS, size=32 * 1024, dtype=np.uint8)
    hist = np.zeros(BINS, dtype=np.uint32)
    _launch_shared(shared_histogram_plus_equals, gpu_backend, pixels, hist)
    np.testing.assert_array_equal(hist.astype(np.int64), _expected(pixels))


def test_no_pixel_is_lost_and_none_is_counted_twice(gpu_backend):
    pixels = (np.arange(4096, dtype=np.int32) % BINS).astype(np.uint8)
    hist = np.zeros(BINS, dtype=np.uint32)
    _launch_shared(shared_histogram, gpu_backend, pixels, hist)
    assert int(hist.sum()) == pixels.size
    np.testing.assert_array_equal(hist.astype(np.int64), _expected(pixels))


def test_a_shared_scratch_buffer_without_atomics(gpu_backend):
    """A local buffer is also usable as ordinary per-lane storage."""
    x = np.arange(BINS, dtype=np.int32)
    hist = np.full(BINS, -1, dtype=np.int32)
    _launch_shared(shared_sums, gpu_backend, x, hist)
    np.testing.assert_array_equal(hist, x * 2)


def test_the_local_count_must_be_a_literal(gpu_backend):
    """The size is a compile-time constant, exactly like `__shared__ T name[N]`."""

    @kernel
    def uses_a_global_count(out: np.ndarray):
        work_size(out)
        slots = zeros_per_group(np.float32, BINS)  # a module global, not a literal
        slots[blockIdx][get_local_id(0)] = 0.0
        barrier()
        out[get_global_id(0)] = slots[blockIdx][get_local_id(0)]

    with pytest.raises(SyntaxError) as caught:
        uses_a_global_count[gpu_backend][(BINS, BINS)](np.zeros(BINS, dtype=np.float32))
    assert "constant" in str(caught.value)


def test_a_shared_buffer_must_be_indexed_by_its_group(gpu_backend):
    """`slots[lane]` names a group the work item cannot reach, so it is refused."""

    @kernel
    def wrong_index(pixels: np.ndarray, out: np.ndarray):
        work_size(out)
        lane = get_local_id(0)
        slots = zeros_per_group(np.int32, 256)
        slots[lane] = 1                    # a local id is not a group id
        barrier()
        out[get_global_id(0)] = slots[lane]

    with pytest.raises(SyntaxError) as caught:
        wrong_index[gpu_backend][(BINS, BINS)](np.zeros(BINS, dtype=np.int32),
                                               np.zeros(BINS, dtype=np.int32))
    assert "block" in str(caught.value).lower()


def test_a_shared_buffer_cannot_be_used_without_a_group():
    """Naming the buffer itself (`out[i] = slots`) is refused: it is a mapping on the host."""

    @kernel
    def bare(out: np.ndarray):
        work_size(out)
        slots = zeros_per_group(np.int32, 256)
        out[get_global_id(0)] = slots

    with pytest.raises(SyntaxError) as caught:
        bare["cuda"][(BINS, BINS)](np.zeros(BINS, dtype=np.int32))
    assert "block" in str(caught.value).lower()


def test_the_interpreter_backend_refuses_shared_memory():
    """Group-shared memory needs parallel work items; a serial run cannot emulate it."""
    pixels = np.zeros(16, dtype=np.uint8)
    with pytest.raises(SyntaxError) as caught:
        shared_histogram["python"](pixels, np.zeros(BINS, dtype=np.uint32))
    assert "zeros_per_group" in str(caught.value)


def test_the_cuda_index_vector_is_the_opencl_scalar(gpu_backend):
    """`blockIdx.x` and `get_group_id(0)` are the same number, spelled two ways."""

    @kernel
    def both(out: np.ndarray):
        work_size(out)
        i = get_global_id(0)
        out[i] = 1 if blockIdx.x == get_group_id(0) else 0

    out = np.zeros(BINS, dtype=np.int32)
    both[gpu_backend][(BINS, BINS)](out)
    np.testing.assert_array_equal(out, np.ones(BINS, dtype=np.int32))


def test_a_local_size_reports_the_group_width(gpu_backend):
    """`get_local_size(0)` is blockDim.x, available for indexing a local buffer."""

    @kernel
    def report(out: np.ndarray):
        work_size(out)
        i = get_global_id(0)
        out[i] = get_local_size(0)

    out = np.zeros(BINS, dtype=np.int32)
    report[gpu_backend][(BINS, BINS)](out)
    np.testing.assert_array_equal(out, np.full(BINS, BINS, dtype=np.int32))
