"""An array the kernel writes only partly keeps its other elements.

Python hands the kernel the caller's array, so an element the body never assigns still
holds what the caller put there. A GPU launch has to reproduce that: the read-back copies
the whole device buffer into the array, so the buffer must have been uploaded first --
otherwise every element the kernel did not write comes back as the zero-initialised
contents of a freshly allocated device buffer.

This is what makes the common "adjust one channel of an RGB image" kernel behave like its
python equivalent, and it is why a write-only argument is uploaded even though the body
never reads it.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def write_first_channel(image: np.ndarray, gain: float):
    """Writes channel 0 of every pixel; channels 1 and 2 are never assigned."""
    for i, j in np.ndindex(image.shape[0], image.shape[1]):
        image[i, j, 0] = image[i, j, 0] * gain
    return None


@kernel
def write_one_element(x: np.ndarray, out: np.ndarray):
    """Writes a single element of a larger output array, one work item per slot."""
    for i in range(out.shape[0]):
        if i == 1:
            out[i] = x[0]
    return None


def test_untouched_channels_survive_a_partial_write(runner):
    image = np.ones((2, 3, 3), dtype=np.float32)
    image[:, :, 0] = 5.0
    runner(write_first_channel, image, 2.0)

    np.testing.assert_allclose(image[:, :, 0], np.full((2, 3), 10.0, dtype=np.float32))
    # The other two channels were never assigned, so python would leave them alone.
    np.testing.assert_allclose(image[:, :, 1:], np.ones((2, 3, 2), dtype=np.float32))


def test_untouched_elements_survive_a_partial_write(runner):
    x = np.array([7.5], dtype=np.float32)
    out = np.full(4, 3.0, dtype=np.float32)
    runner(write_one_element, x, out)

    expected = np.full(4, 3.0, dtype=np.float32)
    expected[1] = 7.5
    np.testing.assert_allclose(out, expected)


@kernel
def fill_everything(out: np.ndarray, value: float):
    """A body that writes every element: the upload changes nothing, but must not break it."""
    for i in range(out.shape[0]):
        out[i] = value
    return None


def test_a_full_write_still_produces_the_new_values(runner):
    out = np.full(4, 3.0, dtype=np.float32)
    runner(fill_everything, out, 9.0)
    np.testing.assert_allclose(out, np.full(4, 9.0, dtype=np.float32))


@kernel
def read_and_write(x: np.ndarray, out: np.ndarray):
    """Ordinary in-place update: read and write the same buffer."""
    for i in range(out.shape[0]):
        out[i] = out[i] + x[i]
    return None


def test_an_in_place_update_keeps_the_previous_contents(runner):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    runner(read_and_write, x, out)
    np.testing.assert_allclose(out, np.array([11.0, 22.0, 33.0], dtype=np.float32))


@kernel
def write_a_created_array(image: np.ndarray):
    """The same rule applies to an array the body created: it starts with its own values."""
    scratch: np.ndarray = np.ones_like(image)
    for i, j in np.ndindex(scratch.shape[0], scratch.shape[1]):
        scratch[i, j, 0] = image[i, j, 0] * 2.0
    return scratch


def test_a_created_array_keeps_the_values_the_body_did_not_write(runner):
    image = np.full((2, 3, 3), 5, dtype=np.float32)
    result = runner(write_a_created_array, image)

    expected = np.ones((2, 3, 3), dtype=np.float32)
    expected[:, :, 0] = image[:, :, 0] * 2.0
    np.testing.assert_allclose(result, expected)
