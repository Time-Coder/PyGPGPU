"""`+=` is atomic exactly where it has to be.

Python accumulates every contribution of `bins[value] += 1`, so device code that emits a
plain read-modify-write loses most of them with no error at all: on a 64x64x3 image a
histogram written that way counted 256 pixels per channel instead of 4096, because every
work item read the same old value and wrote back the same new one.

The translator therefore atomicizes an augmented assignment on an array element whenever it
cannot prove the elements are distinct. It *can* prove that when the subscript is built
from the magic loop's own index variables: distinct work items then have distinct index
tuples, and a plain store is both exact and faster. These tests pin both halves -- the
correctness of the racy case and the absence of atomics in the provably-safe case.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# Racy: the index comes from data, so several work items share a bin
# ---------------------------------------------------------------------------


@kernel
def hist_with_plus_equals(image: np.ndarray):
    bins: np.ndarray = np.zeros((3, 256), dtype=np.int32)
    for i, j, k in np.ndindex(image.shape):
        bins[k, image[i, j, k]] += 1
    return bins


@kernel
def reduce_into_one_slot(x: np.ndarray, out: np.ndarray):
    """Every work item adds into the same element."""
    for i in range(x.shape[0]):
        out[0] += x[i]
    return None


@kernel
def count_above(x: np.ndarray, out: np.ndarray):
    """A single bin incremented by many work items, with -= alongside it."""
    for i in range(x.shape[0]):
        if x[i] > 0.5:
            out[0] += 1
        else:
            out[1] -= 1
    return None


def test_a_data_driven_index_accumulates_every_contribution(runner):
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    expected = np.stack([np.histogram(image[:, :, c], bins=256, range=(0, 256))[0]
                         for c in range(3)]).astype(np.int32)
    np.testing.assert_array_equal(runner(hist_with_plus_equals, image), expected)


def test_a_reduction_into_one_slot_is_exact(runner):
    x = np.ones(64, dtype=np.int32)
    out = np.zeros(1, dtype=np.int32)
    runner(reduce_into_one_slot, x, out)
    np.testing.assert_array_equal(out, np.array([64], dtype=np.int32))


def test_plus_and_minus_both_accumulate(runner):
    x = np.array([0.9, 0.1, 0.8, 0.2, 0.7], dtype=np.float32)
    out = np.zeros(2, dtype=np.int32)
    runner(count_above, x, out)
    np.testing.assert_array_equal(out, np.array([3, -2], dtype=np.int32))


# ---------------------------------------------------------------------------
# Provably safe: the index is the loop's own, so no atomic is emitted
# ---------------------------------------------------------------------------


@kernel
def accumulate_own_element(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] += x[i]
    return None


@kernel
def accumulate_own_element_2d(x: np.ndarray, out: np.ndarray):
    for i, j in np.ndindex(out.shape):
        out[i, j] += x[i, j]
    return None


def test_each_work_item_accumulating_its_own_element(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.full(4, 10.0, dtype=np.float32)
    runner(accumulate_own_element, x, out)
    np.testing.assert_allclose(out, np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32))


def test_a_float_accumulation_still_works_when_it_is_disjoint(runner):
    """No atomic exists for floats, so this would fail if the rule were blanket."""
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    out = np.full((2, 3), 0.5, dtype=np.float64)
    runner(accumulate_own_element_2d, x, out)
    np.testing.assert_allclose(out, x + 0.5)


def _generated_source(kernel_object, backend: str) -> str:
    """The device source this kernel generated on ``backend``, if it has been compiled."""
    import pathlib

    folders = {"opencl": ("__clcache__", ".cl"), "cuda": ("__cucache__", ".cu")}
    folder, suffix = folders[backend]
    cache = pathlib.Path(__file__).parent / folder / "pykernels"
    pattern = f"pyk_{kernel_object.__name__}_*{suffix}"
    for path in sorted(cache.glob(pattern)):
        if "_clean" in path.name:
            continue
        return path.read_text(encoding="utf-8")
    return ""


@pytest.mark.parametrize("backend", ["opencl", "cuda"])
def test_the_safe_case_emits_no_atomic(gpu_backend, runner, backend):
    if backend != gpu_backend:
        pytest.skip(f"generated code is checked on the {backend} backend")
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(accumulate_own_element, x, out)
    source: str = _generated_source(accumulate_own_element, backend)
    assert source, f"no generated source found for {backend}"
    assert "atomic" not in source.lower()


@pytest.mark.parametrize("backend", ["opencl", "cuda"])
def test_the_racy_case_does_emit_an_atomic(gpu_backend, runner, backend):
    if backend != gpu_backend:
        pytest.skip(f"generated code is checked on the {backend} backend")
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    runner(hist_with_plus_equals, image)
    source: str = _generated_source(hist_with_plus_equals, backend)
    assert source, f"no generated source found for {backend}"
    assert "atomic" in source.lower()


# ---------------------------------------------------------------------------
# The racy shapes that cannot be atomicized must be refused, not miscompiled
# ---------------------------------------------------------------------------


@kernel
def racy_float_sum(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        out[0] += x[i]
    return None


@kernel
def racy_multiply(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        out[0] *= 2
    return None


@kernel
def racy_divide(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        out[0] /= 2.0
    return None


def test_a_racy_float_accumulation_is_refused(gpu_backend, runner):
    """There is no float atomic, so this cannot be made exact -- say so."""
    x = np.ones(4, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        runner(racy_float_sum, x, out)
    message: str = str(caught.value)
    assert "atomic" in message
    assert "reduce" in message


def test_an_operator_without_an_atomic_is_refused(gpu_backend, runner):
    x = np.ones(4, dtype=np.int32)
    out = np.ones(1, dtype=np.int32)
    with pytest.raises(SyntaxError) as caught:
        runner(racy_multiply, x, out)
    assert "*=" in str(caught.value)


def test_a_racy_float_division_is_refused(gpu_backend, runner):
    x = np.ones(4, dtype=np.float32)
    out = np.ones(1, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        runner(racy_divide, x, out)
    assert "/=" in str(caught.value)


# ---------------------------------------------------------------------------
# The python backend runs the body as written, so it must agree
# ---------------------------------------------------------------------------


def test_the_python_backend_agrees_on_the_racy_case(runner):
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
    expected = np.stack([np.histogram(image[:, :, c], bins=256, range=(0, 256))[0]
                         for c in range(3)]).astype(np.int32)
    np.testing.assert_array_equal(hist_with_plus_equals["python"](image), expected)
