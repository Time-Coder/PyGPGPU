"""np.clip and np.where on a single element, in each backend's native spelling.

These two were reachable from a kernel body -- `np.clip(alpha * image[i, j, k], 0, 255)`
is the natural way to write a contrast adjustment -- but neither was implemented, so the
translator rejected them. Both now compile into device expressions rather than allocating.

The distinction that matters here: the same numpy function *can* allocate when handed
arrays, so below the magic loop it must still be refused as a creation, while the
element form has to run on the device. The tests check both sides of that line.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# np.clip: both bounds, one bound, integer element
# ---------------------------------------------------------------------------


@kernel
def clip_both_bounds(image: np.ndarray):
    dest: np.ndarray = np.zeros_like(image)
    for i, j, k in np.ndindex(image.shape):
        dest[i, j, k] = np.clip(image[i, j, k] * 2.0, 0, 255)
    return dest


@kernel
def clip_lower_bound_only(image: np.ndarray):
    """A literal None means "no upper bound", exactly as np.clip(a, lo, None) does."""
    dest: np.ndarray = np.zeros_like(image)
    for i, j, k in np.ndindex(image.shape):
        dest[i, j, k] = np.clip(image[i, j, k], 10, None)
    return dest


@kernel
def clip_upper_bound_only(image: np.ndarray):
    dest: np.ndarray = np.zeros_like(image)
    for i, j, k in np.ndindex(image.shape):
        dest[i, j, k] = np.clip(image[i, j, k], None, 100)
    return dest


@kernel
def clip_integer_element(values: np.ndarray):
    """An integer input stays an integer: np.clip(300, 0, 100) is 100, not 100.0."""
    dest = np.zeros_like(values)
    for i in range(values.shape[0]):
        dest[i] = np.clip(values[i], 0, 100)
    return dest


def test_clip_with_both_bounds(runner):
    image = np.array([[[10, 200], [30, 40]]], dtype=np.uint8)
    result = runner(clip_both_bounds, image)
    expected = np.clip(image.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
    np.testing.assert_allclose(result, expected)


def test_clip_with_only_a_lower_bound(runner):
    image = np.array([[[5, 200], [30, 40]]], dtype=np.uint8)
    result = runner(clip_lower_bound_only, image)
    np.testing.assert_allclose(result, np.clip(image, 10, None))


def test_clip_with_only_an_upper_bound(runner):
    image = np.array([[[5, 200], [30, 40]]], dtype=np.uint8)
    result = runner(clip_upper_bound_only, image)
    np.testing.assert_allclose(result, np.clip(image, None, 100))


def test_clip_keeps_an_integer_an_integer(runner):
    values = np.array([5, 500, 50], dtype=np.int32)
    result = runner(clip_integer_element, values)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, np.clip(values, 0, 100))


# ---------------------------------------------------------------------------
# np.where
# ---------------------------------------------------------------------------


@kernel
def where_chooses_a_value(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.where(x[i] > 0.5, x[i], -1.0)
    return None


@kernel
def where_with_integer_values(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.where(x[i] > 0, 1, 0)
    return None


def test_where_selects_between_two_values(runner):
    x = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(where_chooses_a_value, x, out)
    np.testing.assert_allclose(out, np.where(x > 0.5, x, -1.0))


def test_where_with_integer_results(runner):
    x = np.array([-1.0, 2.0, 0.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.int32)
    runner(where_with_integer_values, x, out)
    np.testing.assert_array_equal(out, np.where(x > 0, 1, 0))


# ---------------------------------------------------------------------------
# The array form is still a host creation, so below the loop it is refused
# ---------------------------------------------------------------------------


@kernel
def clip_an_array_below_the_loop(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        scratch = np.clip(x, 0.0, 1.0)
        out[i] = scratch[0]


def test_clipping_a_whole_array_below_the_loop_is_rejected(gpu_backend, runner):
    """`np.clip(v, lo, hi)` stores a value; `np.clip(arr, lo, hi)` allocates."""
    with pytest.raises(SyntaxError) as caught:
        runner(clip_an_array_below_the_loop, np.zeros(2, np.float32), np.zeros(2, np.float32))
    assert "magic loop" in str(caught.value)


@kernel
def clip_an_array_element_expression(x: np.ndarray, out: np.ndarray):
    """The element form is device code even though the function name can also allocate."""
    for i in range(out.shape[0]):
        out[i] = np.clip(x[i], 0.0, 1.0)
    return None


def test_the_element_form_is_device_code(runner):
    x = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(clip_an_array_element_expression, x, out)
    np.testing.assert_allclose(out, np.clip(x, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Each backend uses its own native spelling
# ---------------------------------------------------------------------------


@kernel
def clip_for_codegen(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.clip(x[i], 0.0, 1.0)
    return None


def _generated_source(backend: str) -> str:
    """The device source this kernel generates on ``backend``, if it has been compiled."""
    import pathlib

    folders = {"opencl": ("__clcache__", ".cl"), "cuda": ("__cucache__", ".cu"),
               "hip": ("__hipcache__", ".hip")}
    folder, suffix = folders[backend]
    cache = pathlib.Path(__file__).parent / folder / "pykernels"
    for path in sorted(cache.glob(f"pyk_clip_for_codegen_*{suffix}")):
        if "_clean" in path.name:
            continue
        return path.read_text(encoding="utf-8")
    return ""


@pytest.mark.parametrize("backend", ["opencl", "cuda"])
def test_clip_uses_the_backend_native_form(gpu_backend, runner, backend):
    """OpenCL has a real `clamp` builtin; CUDA does not, so it needs min/max."""
    if backend != gpu_backend:
        pytest.skip(f"generated code is checked on the {backend} backend")
    x = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(clip_for_codegen, x, out)
    source: str = _generated_source(backend)
    assert source, f"no generated source found for {backend}"
    if backend == "opencl":
        assert "clamp(" in source
        assert "fmin" not in source
    else:
        assert "clamp(" not in source
        assert "fmin(" in source and "fmax(" in source
