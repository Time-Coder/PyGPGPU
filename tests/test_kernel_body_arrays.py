"""Arrays created *inside* the work-item body.

A statement above the magic loop runs once per launch, so its object travels as a parameter
(the host builds it). A statement below the loop runs once per work item, so it cannot be a
host object at all: it is private storage for each work item, which is what a C local array
is. Python creates a fresh array every iteration, so the device does too.

The shape has to be a compile-time constant -- a private array is a fixed-size declaration --
and anything else is refused with a message rather than silently shared between work items.
"""
from __future__ import annotations

import numpy as np
import pytest
from pygf import float3

from pygpgpu import kernel

PAIR = np.dtype([("i", "<i4"), ("f", "<f4")])


# ---------------------------------------------------------------------------
# A scalar array created in the body
# ---------------------------------------------------------------------------


@kernel
def scalar_array_in_body(out: np.ndarray):
    for i in range(out.shape[0]):
        scratch: np.ndarray = np.zeros(4, dtype=np.float32)
        scratch[1] = 5.0
        scratch[2] = scratch[1] + 1.0
        out[i] = scratch[2]


def test_a_scalar_array_created_in_the_body(runner):
    out = np.zeros(2, dtype=np.float32)
    runner(scalar_array_in_body, out)
    np.testing.assert_allclose(out, np.full(2, 6.0, dtype=np.float32))


@kernel
def array_in_body_is_private(out: np.ndarray):
    """Every work item has its own copy: one item's writes must not reach another's."""
    for i in range(out.shape[0]):
        scratch: np.ndarray = np.zeros(2, dtype=np.float32)
        scratch[0] = float(i) + 1.0
        out[i] = scratch[0]


def test_the_array_is_private_per_work_item(runner):
    out = np.zeros(4, dtype=np.float32)
    runner(array_in_body_is_private, out)
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# The other element types
# ---------------------------------------------------------------------------


@kernel
def ones_and_full(out: np.ndarray):
    for i in range(out.shape[0]):
        ones: np.ndarray = np.ones(3, dtype=np.int32)
        filled: np.ndarray = np.full(3, 7, dtype=np.int32)
        out[i] = ones[1] + filled[2]


def test_ones_and_full_initialise_the_private_array(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(ones_and_full, out)
    assert float(out[0]) == pytest.approx(8.0)


@kernel
def vector_array_in_body(out: np.ndarray):
    for i in range(out.shape[0]):
        scratch: np.ndarray = np.zeros(2, dtype=float3.numpy_dtype())
        scratch[0] = float3(1.0, 2.0, 3.0)
        # A vector element is a vector, so its components are attributes (a structure's
        # members are string keys).
        out[i] = scratch[0].z


def test_a_vector_array_created_in_the_body(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(vector_array_in_body, out)
    assert float(out[0]) == pytest.approx(3.0)


@kernel
def struct_array_in_body(out: np.ndarray):
    for i in range(out.shape[0]):
        scratch: np.ndarray = np.zeros(2, dtype=PAIR)
        scratch[1]["i"] = 9
        out[i] = scratch[1]["i"]


def test_a_struct_array_created_in_the_body(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(struct_array_in_body, out)
    assert float(out[0]) == pytest.approx(9.0)


@kernel
def two_dimensional_array_in_body(out: np.ndarray):
    for i in range(out.shape[0]):
        grid: np.ndarray = np.zeros((2, 3), dtype=np.float32)
        grid[1, 2] = 4.0
        out[i] = grid[0, 0] + grid[1, 2]


def test_a_two_dimensional_array_created_in_the_body(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(two_dimensional_array_in_body, out)
    assert float(out[0]) == pytest.approx(4.0)


@kernel
def shape_of_a_created_array(out: np.ndarray):
    for i in range(out.shape[0]):
        grid: np.ndarray = np.zeros((2, 3), dtype=np.float32)
        out[i] = grid.shape[0] * 10 + grid.shape[1]


def test_the_shape_of_a_created_array(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(shape_of_a_created_array, out)
    assert float(out[0]) == pytest.approx(23.0)


# ---------------------------------------------------------------------------
# What cannot work is refused, not silently shared
# ---------------------------------------------------------------------------


@kernel
def created_from_an_argument_shape(x: np.ndarray, out: np.ndarray):
    """The extent is a runtime value, so a fixed-size private array cannot hold it."""
    for i in range(out.shape[0]):
        scratch: np.ndarray = np.zeros(x.shape[0], dtype=np.float32)
        out[i] = scratch[0]


def test_an_array_with_a_runtime_shape_is_refused(gpu_backend, runner):
    out = np.zeros(1, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        runner(created_from_an_argument_shape, np.zeros(4, dtype=np.float32), out)
    message: str = str(caught.value)
    assert "constant" in message or "compile" in message
