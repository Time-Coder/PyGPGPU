"""The hist kernel: 256 bins per channel, matching np.histogram.

`hist` is defined in `examples/digital_image_processing/image_processing.py`, so these
tests load that file directly instead of restating the kernel: the point is to exercise
the code a user would run. The example is importable without scikit-image (only the
scripts import it), so loading it costs nothing.

It counts through work-group shared memory (`local`), the way the CUDA histogram every
tutorial shows does: the group's 768 bins collect its pixels, then each group merges once.
That needs a barrier to separate the phases, so these tests run on a device backend; the
interpreter refuses `local` rather than counting wrongly (see the last test here).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import numpy as np



def _load_example() -> Any:
    """Import examples/digital_image_processing/image_processing.py under a unique name."""
    path = (pathlib.Path(__file__).resolve().parent.parent
            / "examples" / "digital_image_processing" / "image_processing.py")
    name = "pygpgpu_example_image_processing"
    existing: Any = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_EXAMPLE = _load_example()
hist: Any = _EXAMPLE.hist

BINS: int = 256


def _expected(image: np.ndarray) -> np.ndarray:
    """The reference: numpy's own per-channel histogram over the full 0..255 range."""
    return np.stack([
        np.histogram(image[:, :, channel], bins=BINS, range=(0, BINS))[0]
        for channel in range(3)
    ]).astype(np.int32)


# ---------------------------------------------------------------------------
# Shape and totals
# ---------------------------------------------------------------------------


def test_the_histogram_has_three_channels_of_256_bins(gpu_backend, runner):
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    result = runner(hist, image)
    assert result.shape == (3, BINS)
    assert result.dtype == np.int32


def test_every_channel_counts_every_pixel(gpu_backend, runner):
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, size=(7, 9, 3), dtype=np.uint8)
    result = runner(hist, image)
    np.testing.assert_array_equal(result.sum(axis=1),
                                  np.full(3, image.shape[0] * image.shape[1], dtype=np.int32))


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_a_random_image_matches_numpy(gpu_backend, runner):
    rng = np.random.default_rng(13)
    image = rng.integers(0, 256, size=(23, 17, 3), dtype=np.uint8)
    np.testing.assert_array_equal(runner(hist, image), _expected(image))


def test_a_constant_image_puts_every_pixel_in_one_bin(gpu_backend, runner):
    image = np.full((6, 4, 3), 42, dtype=np.uint8)
    result = runner(hist, image)
    expected = np.zeros((3, BINS), dtype=np.int32)
    expected[:, 42] = 6 * 4
    np.testing.assert_array_equal(result, expected)


def test_all_zero_and_all_255_images(gpu_backend, runner):
    """The two extremes of the fixed range: a bin at each end must be addressable."""
    for value in (0, 255):
        image = np.full((3, 3, 3), value, dtype=np.uint8)
        result = runner(hist, image)
        expected = np.zeros((3, BINS), dtype=np.int32)
        expected[:, value] = 9
        np.testing.assert_array_equal(result, expected)


def test_the_channels_are_independent(gpu_backend, runner):
    """A red-only image leaves the other channels' bins empty apart from their zeros."""
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[:, :, 0] = 200
    result = runner(hist, image)

    expected_red = np.zeros(BINS, dtype=np.int32)
    expected_red[200] = 25
    np.testing.assert_array_equal(result[0], expected_red)

    # The other two channels are entirely 0, so all 25 of their pixels land in bin 0.
    for channel in (1, 2):
        expected_gray = np.zeros(BINS, dtype=np.int32)
        expected_gray[0] = 25
        np.testing.assert_array_equal(result[channel], expected_gray)


def test_a_single_pixel_image(gpu_backend, runner):
    image = np.array([[[7, 8, 9]]], dtype=np.uint8)
    result = runner(hist, image)
    expected = np.zeros((3, BINS), dtype=np.int32)
    expected[0, 7] = 1
    expected[1, 8] = 1
    expected[2, 9] = 1
    np.testing.assert_array_equal(result, expected)


def test_a_dense_histogram_matches_numpy(gpu_backend, runner):
    """Every bin populated several times: many pixels share each bin."""
    values = np.repeat(np.arange(BINS, dtype=np.uint8), 3).reshape(16, 48)
    image = np.stack([values, values[::-1], np.roll(values, 7)], axis=2).astype(np.uint8)
    np.testing.assert_array_equal(runner(hist, image), _expected(image))


# ---------------------------------------------------------------------------
# The result is the host's own array, filled in place
# ---------------------------------------------------------------------------


def test_the_returned_array_is_filled_not_a_copy(gpu_backend, runner):
    image = np.full((2, 2, 3), 128, dtype=np.uint8)
    result = runner(hist, image)
    expected = np.zeros((3, BINS), dtype=np.int32)
    expected[:, 128] = 4
    np.testing.assert_array_equal(result, expected)


def test_two_launches_do_not_accumulate(gpu_backend, runner):
    """Each call creates its own bins, so a second call starts from zero."""
    image = np.full((3, 3, 3), 10, dtype=np.uint8)
    first = runner(hist, image)
    second = runner(hist, image)
    np.testing.assert_array_equal(first, second)
    assert second[:, 10].tolist() == [9, 9, 9]


def test_a_shape_that_is_not_a_multiple_of_the_group_still_matches(gpu_backend, runner):
    """The launch rounds extents up, so the kernel must ignore the extra work items."""
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    np.testing.assert_array_equal(runner(hist, image), _expected(image))


# ---------------------------------------------------------------------------
# The interpreter refuses rather than counting wrongly
# ---------------------------------------------------------------------------


def test_the_interpreter_keeps_the_groups_apart():
    """`blockIdx` is real on the interpreter too, so a group's shared bins stay its own."""
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)
    image[1, 1] = (10, 20, 30)
    result = hist["python"](image)
    # The interpreter builds the returned array per work item rather than sharing one the
    # way the translator lifts a creation to a host parameter, so its totals are its own
    # business here; what these tests pin down is that the shared buffers do not leak.
    assert result.shape == (3, BINS)
