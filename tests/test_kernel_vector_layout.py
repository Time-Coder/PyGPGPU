"""Vector layout across the host/device boundary.

The host (`pygf`) stores a 3-component vector in three components. The device ABI is not
uniform: CUDA/HIP also use three, OpenCL pads to four (`cl_float3` is 16 bytes). So a
3-component vector -- and anything that contains one -- has to be re-laid-out on the way into
an OpenCL kernel and on the way back, while staying untouched for CUDA/HIP.

`pygpgpu.vec_types.layout` is that boundary. These tests pin the primitives and the four
places a vector crosses it: an argument, a returned value, an outer variable, and an object
the body created.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest
from pygf import float3, float4, int3

from pygpgpu import kernel
from pygpgpu.vec_types.layout import _host_classes, _vector_dtype_name, pack


# ---------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------


def test_the_host_classes_are_the_pygf_ones():
    """A 3-component vector is three components on the host, so `sizeof` is 12."""
    assert ctypes.sizeof(_host_classes()["float3"]) == 12
    assert [f for f, _ in _host_classes()["float3"]._fields_] == ["x", "y", "z"]


def test_a_dtype_is_told_apart_by_its_types_not_its_names():
    """`float3` and `int3` share the names x/y/z; only the component type differs."""
    assert _vector_dtype_name(float3.numpy_dtype()) == "float3"
    assert _vector_dtype_name(int3.numpy_dtype()) == "int3"
    assert _vector_dtype_name(float4.numpy_dtype()) == "float4"
    assert _vector_dtype_name(np.dtype("<f4")) is None


def test_opencl_packs_a_three_component_vector_to_four_slots():
    packed, unpack = pack(float3(1.0, 2.0, 3.0), "opencl")
    assert ctypes.sizeof(packed) == 16
    assert [f for f, _ in type(packed)._fields_] == ["x", "y", "z", "_"]
    assert (packed.x, packed.y, packed.z) == (1.0, 2.0, 3.0)


def test_cuda_keeps_the_host_layout():
    original = float3(1.0, 2.0, 3.0)
    packed, unpack = pack(original, "cuda")
    assert packed is original
    assert unpack is None


def test_a_four_component_vector_needs_no_padding():
    original = float4(1.0, 2.0, 3.0, 4.0)
    packed, unpack = pack(original, "opencl")
    assert packed is original and unpack is None


def test_an_array_of_three_component_vectors_is_repacked_and_restored():
    host = np.zeros(2, dtype=float3.numpy_dtype())
    host[:] = (1.0, 2.0, 3.0)
    packed, unpack = pack(host, "opencl")
    assert packed.dtype.itemsize == 16 and host.dtype.itemsize == 12
    packed["x"] = [7.0, 8.0]
    packed["_"] = [-1.0, -1.0]
    unpack(packed)
    assert host["x"].tolist() == [7.0, 8.0]
    assert host["y"].tolist() == [2.0, 2.0]


class _WithVector(ctypes.Structure):
    _fields_ = [("v", float3), ("k", ctypes.c_int), ("xs", float3 * 2)]


def test_a_structure_holding_vectors_is_relaid_out_memberwise():
    host = _WithVector()
    host.v = float3(1.0, 2.0, 3.0)
    host.k = 5
    host.xs[0] = float3(4.0, 5.0, 6.0)

    packed, unpack = pack(host, "opencl")
    assert ctypes.sizeof(packed) > ctypes.sizeof(host)
    assert ctypes.sizeof(packed.v) == 16
    assert (packed.v.x, packed.xs[0].z) == (1.0, 6.0)
    assert packed.k == 5

    packed.v = type(packed.v)(9.0, 8.0, 7.0)
    unpack(packed)
    assert (host.v.x, host.v.y, host.v.z) == (9.0, 8.0, 7.0)


# ---------------------------------------------------------------------------
# The boundary, through a real launch
# ---------------------------------------------------------------------------


@kernel
def read_one_vector(v: float3, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = v.x + v.y + v.z


@kernel
def read_vector_array(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i].x


@kernel
def write_vector_array(a: np.ndarray):
    for i in range(a.shape[0]):
        a[i] = float3(float(i) + 1.0, 2.0, 3.0)
    return a


@kernel
def return_one_vector(x: np.ndarray):
    for i in range(x.shape[0]):
        return float3(1.0, 2.0, 3.0)


OUTER_VECTORS: np.ndarray = np.zeros(2, dtype=float3.numpy_dtype())


@kernel
def read_outer_vector(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = OUTER_VECTORS[0].y


@kernel
def create_vector_array(out: np.ndarray):
    scratch: np.ndarray = np.zeros(2, dtype=float3.numpy_dtype())
    scratch[0] = float3(7.0, 8.0, 9.0)
    for i in range(out.shape[0]):
        out[i] = scratch[0].z
    return scratch


def test_a_vector_argument(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(read_one_vector, float3(1.0, 2.0, 3.0), out)
    assert float(out[0]) == pytest.approx(6.0)


def test_a_three_component_vector_array_is_readable(runner):
    host = np.zeros(2, dtype=float3.numpy_dtype())
    host[:] = (5.0, 0.0, 0.0)
    out = np.zeros(1, dtype=np.float32)
    runner(read_vector_array, host, out)
    assert float(out[0]) == 5.0


def test_a_written_vector_array_comes_back_in_the_host_layout(runner):
    """The kernel writes the device layout; the caller's array must still be 3 components."""
    host = np.zeros(2, dtype=float3.numpy_dtype())
    result = runner(write_vector_array, host)
    assert result is host
    assert host.dtype.itemsize == 12
    assert host["x"].tolist() == [1.0, 2.0]
    assert host["z"].tolist() == [3.0, 3.0]


def test_a_returned_vector_is_a_vector_instance(runner):
    returned = runner(return_one_vector, np.zeros(1, dtype=np.float32))
    assert isinstance(returned, float3)
    assert (returned.x, returned.y, returned.z) == (1.0, 2.0, 3.0)


def test_an_outer_vector_array_is_visible(runner):
    OUTER_VECTORS[:] = (0.0, 9.0, 0.0)
    out = np.zeros(1, dtype=np.float32)
    runner(read_outer_vector, out)
    assert float(out[0]) == 9.0


def test_a_created_vector_array_travels_and_returns(runner):
    out = np.zeros(1, dtype=np.float32)
    created = runner(create_vector_array, out)
    assert float(out[0]) == 9.0
    assert created.dtype.itemsize == 12
    assert float(created["z"][0]) == 9.0
