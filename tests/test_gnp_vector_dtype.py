"""A vector class as an array dtype: `np.dtype(pygf.float3)`, and kernels over it.

numpy builds a structured dtype from a `ctypes` class, and for a *ctypes-derived* dtype its
own subscript answers an element as a live instance of that class: `arr[0]` is a
`pygf.float3`, with the vector's components and arithmetic. The element is a **snapshot** --
numpy builds it from the element's bytes, so a write through it does not reach the array (a
live view is available through `pygf.float3.from_buffer`). Assigning a vector value into a
slot works the other way round.

Kernels consume such an array like any vector-typed one: the element type is the device's
`float3`, so `a[i] * 2.0` is a vector multiply, `a[i].y` a component read, and a kernel that
writes through the argument takes the 3-component unpack path on the way back. A gnp array of
the same dtype reads and writes its elements through the same class (gnp reads elements
through the raw `ndarray.__getitem__` slot, which answers a plain void -- the class is
recovered from the dtype and the element rebuilt, snapshot semantics again), and a kernel
launched on a gnp array of this dtype reads and writes it like any other.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import pygf
from pygpgpu import kernel
from pygpgpu import numpy as gnp


@kernel
def scales(a, out):
    for i in range(out.shape[0]):
        out[i] = a[i] * 2.0


@kernel
def reads_components(a, out):
    for i in range(out.shape[0]):
        out[i] = a[i].y


@kernel
def adds(a, b, out):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


@kernel
def copies(a, out):
    for i in range(out.shape[0]):
        out[i] = a[i]


@kernel
def stores_components(a, out):
    for i in range(out.shape[0]):
        out[i].x = a[i].z
        out[i].z = a[i].x


def _filled() -> np.ndarray:
    arr = np.zeros(4, dtype=pygf.float3.dtype)
    for i, values in enumerate(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0),
                                (7.0, 8.0, 9.0), (10.0, 11.0, 12.0))):
        arr[i] = values
    return arr


def test_the_vector_class_is_a_numpy_dtype():
    dtype = np.dtype(pygf.float3)
    assert dtype.names == ("x", "y", "z")
    assert dtype.itemsize == ctypes.sizeof(pygf.float3)


def test_an_element_is_a_live_vector_value():
    """`arr[0]` is a `pygf.float3`: components, arithmetic, and vector results."""
    arr = _filled()
    element = arr[0]
    assert isinstance(element, pygf.float3)
    assert element.x == pytest.approx(1.0) and element.z == pytest.approx(3.0)
    total = element + arr[1]
    assert isinstance(total, pygf.float3)
    assert (total.x, total.y, total.z) == pytest.approx((5.0, 7.0, 9.0))
    assert (element * 2.0).y == pytest.approx(4.0)


def test_an_element_is_a_snapshot_like_numpy_makes_it():
    """A write through the element does not reach the array -- numpy's own rule."""
    arr = _filled()
    arr[0].x = 42.0
    assert arr[0].x == pytest.approx(1.0)
    # The store spelling, and a live view for the cases that want one.
    arr[0] = pygf.float3(9.0, 8.0, 7.0)
    assert (arr[0].x, arr[0].y, arr[0].z) == pytest.approx((9.0, 8.0, 7.0))
    view = pygf.float3.from_buffer(arr, arr.itemsize)
    view.y = 5.0
    assert arr[1].y == pytest.approx(5.0)


def test_the_monkey_patched_array_builds_the_dtype():
    values = np.array([pygf.float3(1.0, 2.0, 3.0), pygf.float3(4.0, 5.0, 6.0)])
    assert values.dtype == pygf.float3.dtype
    assert isinstance(values[1], pygf.float3)


def test_kernels_read_the_element_as_a_vector(runner):
    """`a[i] * 2.0` is a vector multiply; `a[i].y` a component read; `a[i]` a copy."""
    arr = _filled()
    scaled = np.zeros(4, dtype=pygf.float3.dtype)
    runner(scales, arr, scaled)
    assert (scaled[0].x, scaled[0].z) == pytest.approx((2.0, 6.0))

    components = np.zeros(4, dtype=np.float32)
    runner(reads_components, arr, components)
    np.testing.assert_allclose(components, [2.0, 5.0, 8.0, 11.0])

    copied = np.zeros(4, dtype=pygf.float3.dtype)
    runner(copies, arr, copied)
    assert (copied[2].x, copied[2].z) == pytest.approx((7.0, 9.0))


def test_kernels_add_elements(runner):
    arr = _filled()
    total = np.zeros(4, dtype=pygf.float3.dtype)
    runner(adds, arr, arr, total)
    assert (total[1].x, total[1].y, total[1].z) == pytest.approx((8.0, 10.0, 12.0))


def test_a_kernel_stores_components(gpu_backend):
    """`out[i].x = a[i].z` compiles into a component store on every device backend.

    The store argument is write-only, so the whole-buffer read-back answers the device's
    own y (0): only the two components the kernel wrote come back set. The interpreter
    backend is the exception by construction -- it runs `out[i].x = ...` as python, and
    `out[i]` is numpy's element *snapshot* (a copy), so python itself drops that store.
    """
    arr = _filled()
    swizzled = np.zeros(4, dtype=pygf.float3.dtype)
    stores_components[gpu_backend](arr, swizzled)
    assert (swizzled[0].x, swizzled[0].y, swizzled[0].z) == pytest.approx((3.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# A gnp array of the same dtype
# ---------------------------------------------------------------------------


def test_a_gnp_array_reads_its_elements_through_the_class():
    device = gnp.zeros(4, dtype=pygf.float3.dtype)
    device[0] = pygf.float3(1.0, 2.0, 3.0)
    element = device[0]
    assert isinstance(element, pygf.float3)
    assert (element.x, element.y, element.z) == pytest.approx((1.0, 2.0, 3.0))


def test_a_gnp_element_is_a_snapshot_too(gpu_backend):
    """The gnp element read matches numpy's: a copy, not a live view."""
    device = gnp.zeros(4, dtype=pygf.float3.dtype)
    device[0] = pygf.float3(1.0, 2.0, 3.0)
    element = device[0]
    element.x = 42.0
    assert device[0].x == pytest.approx(1.0)


def test_a_gnp_kernel_reads_and_writes_the_dtype(gpu_backend):
    device = gnp.zeros(4, dtype=pygf.float3.dtype)
    device[0] = pygf.float3(1.0, 2.0, 3.0)
    device[1] = pygf.float3(4.0, 5.0, 6.0)

    scaled = gnp.zeros(4, dtype=pygf.float3.dtype)
    scales[gpu_backend](device, scaled)
    assert (scaled[0].x, scaled[0].z) == pytest.approx((2.0, 6.0))

    components = gnp.zeros(4, dtype=np.float32)
    reads_components[gpu_backend](device, components)
    # The supported host reads: an element, which waits for the kernel's write.
    assert components[0] == pytest.approx(2.0)
    assert components[1] == pytest.approx(5.0)

    other = gnp.zeros(4, dtype=pygf.float3.dtype)
    stores_components[gpu_backend](device, other)
    assert (other[0].x, other[0].y, other[0].z) == pytest.approx((3.0, 0.0, 1.0))
    assert (other[1].x, other[1].y, other[1].z) == pytest.approx((6.0, 0.0, 4.0))
