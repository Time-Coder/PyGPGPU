"""Vectors as kernel arguments and `self` attributes, and host-side scalars as extents.

A ray tracer keeps its state in objects (`self.position: point3f`), so the `@kernel` body
reads vectors that the host owns. Three things had to line up:

* `infer_ctype` recognises a `pygf` vector instance through its class's bases, so a vector
  argument or `self` attribute gets the right device type;
* CUDA/HIP pass a 3-component vector by value in a compact 12-byte layout, and the launch
  needs the *address of the value*, so it is packed into a ctypes buffer rather than an
  array (``numpy.ndarray`` has no storage info to take a pointer to);
* an integer `self` attribute can be the iteration domain:
  ``np.ndindex(self.image_height, self.image_width)``.
"""
from __future__ import annotations

import numpy as np
import pytest
from pygf import color3f, float3, point3f

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# A vector argument
# ---------------------------------------------------------------------------


@kernel
def sum_vector(position: point3f, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = position.x + position.y + position.z


@kernel
def scale_vector(direction: float3, k: float, out: np.ndarray):
    """Vector arithmetic on a parameter: `v * scalar` expands per component."""
    for i in range(out.shape[0]):
        v = direction * k
        out[i] = v.x + v.y + v.z


@kernel
def offset_vector(origin: point3f, scale: float, out: np.ndarray):
    """Two vector parameters in one expression."""
    for i in range(out.shape[0]):
        moved = origin + float3(scale, scale, scale)
        out[i] = moved.x + moved.y + moved.z


def test_a_vector_argument_is_read_component_wise(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(sum_vector, point3f(1.0, 2.0, 3.0), out)
    assert float(out[0]) == pytest.approx(6.0)


def test_vector_arithmetic_on_a_parameter(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(scale_vector, float3(1.0, 2.0, 3.0), 2.0, out)
    assert float(out[0]) == pytest.approx(12.0)


def test_two_vector_parameters(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(offset_vector, point3f(1.0, 2.0, 3.0), 0.5, out)
    assert float(out[0]) == pytest.approx(1.5 + 2.5 + 3.5)


# ---------------------------------------------------------------------------
# A vector as a `self` attribute
# ---------------------------------------------------------------------------


class Scene:
    def __init__(self) -> None:
        self.position = point3f(1.0, 2.0, 3.0)
        self.background = color3f(0.1, 0.2, 0.3)

    @kernel
    def render(self) -> np.ndarray:
        image: np.ndarray = np.zeros((2, 1), dtype=np.float32)
        for i in range(image.shape[0]):
            image[i, 0] = (self.position.x + self.position.y + self.position.z
                           + self.background.r)
        return image

    @kernel
    def scatter(self, direction: float3) -> np.ndarray:
        """A vector attribute and a vector parameter together."""
        image: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        for i in range(image.shape[0]):
            image[i, 0] = ((self.position + direction).x
                           + self.background.g)
        return image


def test_a_vector_self_attribute(backend):
    image = Scene().render[backend]()
    assert float(image[0, 0]) == pytest.approx(6.0 + 0.1)


def test_a_vector_attribute_combined_with_a_parameter(backend):
    image = Scene().scatter[backend](float3(1.0, 0.0, 0.0))
    assert float(image[0, 0]) == pytest.approx(2.0 + 0.2)


# ---------------------------------------------------------------------------
# Host-side integers as the iteration domain
# ---------------------------------------------------------------------------


class Grid:
    def __init__(self, height: int, width: int) -> None:
        self.image_height = height
        self.image_width = width

    @kernel
    def fill(self) -> np.ndarray:
        image: np.ndarray = np.zeros((self.image_height, self.image_width), dtype=np.int32)
        for i, j in np.ndindex(self.image_height, self.image_width):
            image[i, j] = i * 10 + j
        return image


def test_integer_attributes_can_be_the_loop_extents(backend):
    image = Grid(3, 4).fill[backend]()
    expected = np.array([[i * 10 + j for j in range(4)] for i in range(3)], dtype=np.int32)
    np.testing.assert_array_equal(image, expected)


def test_the_extent_follows_the_attribute_value(backend):
    """The launch size is read at call time, not baked in at decoration time."""
    small = Grid(2, 2).fill[backend]()
    large = Grid(5, 3).fill[backend]()
    assert small.shape == (2, 2)
    assert large.shape == (5, 3)


# ---------------------------------------------------------------------------
# Constructing a class stays refused, with a message that says why
# ---------------------------------------------------------------------------


class Canvas:
    """The image is an attribute, the way a renderer keeps its output buffer."""

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.image = np.zeros((height, width), dtype=np.int32)

    @kernel
    def paint(self) -> np.ndarray:
        for i, j in np.ndindex(self.height, self.width):
            self.image[i, j] = i * 10 + j
        return self.image


def test_a_self_array_attribute_can_be_returned(runner):
    """`return self.image` hands back that very array, like python would."""
    canvas = Canvas(3, 4)
    result = runner(canvas.paint)
    expected = np.array([[i * 10 + j for j in range(4)] for i in range(3)], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)
    assert result is canvas.image


class Ray:
    def __init__(self, origin: point3f, direction: float3) -> None:
        self.origin = origin
        self.direction = direction


@kernel
def builds_a_ray(origin: point3f, direction: float3, out: np.ndarray):
    for i in range(out.shape[0]):
        ray = Ray(origin, direction)
        out[i] = ray.origin.x


def test_constructing_a_plain_class_becomes_a_structure(gpu_backend, runner):
    """A plain class a body builds is a layout, decided by what the body does with it.

    The members are the attributes the kernel actually reads (`ray.origin` here), typed from
    the constructor's arguments; `direction` is read by nothing, so the device structure has
    one member and the second argument is dropped -- python's own rule for an attribute
    nobody mentions.
    """
    out = np.zeros(1, dtype=np.float32)
    runner(builds_a_ray, point3f(0.0, 0.0, 0.0), float3(0.0, 0.0, 1.0), out)
    assert float(out[0]) == 0.0
