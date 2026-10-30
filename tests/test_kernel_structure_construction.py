"""Constructing a structure inside a kernel body.

A `ctypes.Structure` subclass is a layout, not an object, so `Ray(origin, direction)` is a
structure *value* -- the same thing `np.zeros(1, dtype=Ray)` holds, without the array around
it. That is what makes the ray-tracing example's `ray = Ray(self.position, ray_dir)` work.

Two things this pins beyond the constructor itself: a helper function whose signature names a
structure needs that typedef emitted too (the type block belongs to the whole source, not to
each function), and a structure built in the body has to read its members by name even in the
interpreter, which runs the body unchanged.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest
from pygf import float3, point3f

from pygpgpu import kernel


class Ray(ctypes.Structure):
    _fields_ = [("origin", point3f), ("direction", float3)]


class Pair(ctypes.Structure):
    _fields_ = [("i", ctypes.c_int), ("f", ctypes.c_float)]


class Inner(ctypes.Structure):
    _fields_ = [("a", ctypes.c_int), ("b", ctypes.c_int)]


class Outer(ctypes.Structure):
    _fields_ = [("inner", Inner), ("c", ctypes.c_float)]


# ---------------------------------------------------------------------------
# The constructor forms
# ---------------------------------------------------------------------------


@kernel
def construct_vector_members(out: np.ndarray):
    for i in range(out.shape[0]):
        ray = Ray(point3f(1.0, 2.0, 3.0), float3(0.0, 0.0, 4.0))
        out[i] = ray["origin"]["x"] + ray["direction"]["z"]


@kernel
def construct_scalar_members(out: np.ndarray):
    for i in range(out.shape[0]):
        pair = Pair(7, 0.5)
        out[i] = pair["i"] + pair["f"]


@kernel
def construct_with_one_member(out: np.ndarray):
    """An omitted member is zero, the way a C aggregate initialiser leaves it."""
    for i in range(out.shape[0]):
        ray = Ray(point3f(5.0, 0.0, 0.0))
        out[i] = ray["origin"]["x"] + ray["direction"]["y"]


@kernel
def construct_each_work_item(out: np.ndarray):
    """One per work item, from that item's own index, like the example's inner loop."""
    for i in range(out.shape[0]):
        ray = Ray(point3f(float(i), 0.0, 0.0), float3(1.0, 0.0, 0.0))
        out[i] = ray["origin"]["x"]


@kernel
def construct_nested(out: np.ndarray):
    for i in range(out.shape[0]):
        outer = Outer(Inner(4, 5), 0.25)
        out[i] = outer["inner"]["b"] + outer["c"]


def test_a_structure_of_vectors(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(construct_vector_members, out)
    assert float(out[0]) == pytest.approx(5.0)


def test_a_structure_of_scalars(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(construct_scalar_members, out)
    assert float(out[0]) == pytest.approx(7.5)


def test_an_omitted_member_is_zero(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(construct_with_one_member, out)
    assert float(out[0]) == pytest.approx(5.0)


def test_one_structure_per_work_item(runner):
    out = np.zeros(3, dtype=np.float32)
    runner(construct_each_work_item, out)
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 2.0], dtype=np.float32))


def test_a_nested_structure_constructor(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(construct_nested, out)
    assert float(out[0]) == pytest.approx(5.25)


# ---------------------------------------------------------------------------
# Too many arguments is a clear error, not a compile failure
# ---------------------------------------------------------------------------


@kernel
def construct_too_many(out: np.ndarray):
    for i in range(out.shape[0]):
        ray = Ray(point3f(1.0, 0.0, 0.0), float3(1.0, 0.0, 0.0), float3(1.0, 0.0, 0.0))
        out[i] = ray["origin"]["x"]


def test_too_many_arguments_is_reported(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(construct_too_many, np.zeros(1, dtype=np.float32))
    assert "at most" in str(caught.value)


# ---------------------------------------------------------------------------
# A helper whose signature names a structure
# ---------------------------------------------------------------------------


def helper_takes_structure(ray, scale):
    """A helper called from the kernel: its parameter is a structure type."""
    return ray["origin"]["x"] * scale


@kernel
def call_helper_with_structure(out: np.ndarray):
    for i in range(out.shape[0]):
        made = Ray(point3f(2.0, 0.0, 0.0), float3(1.0, 0.0, 0.0))
        out[i] = helper_takes_structure(made, 3.0)


def test_a_helper_signature_naming_a_structure(runner):
    """The typedef has to be emitted for the helper too, once, ahead of every function."""
    out = np.zeros(1, dtype=np.float32)
    runner(call_helper_with_structure, out)
    assert float(out[0]) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# The spelling a structure and a vector share
# ---------------------------------------------------------------------------


def test_a_vector_instance_answers_a_named_component():
    """`v["x"]` is the same member the structure subscript reads, so one body works."""
    vector = float3(1.0, 2.0, 3.0)
    assert vector["x"] == pytest.approx(1.0)
    assert vector["z"] == pytest.approx(3.0)
    assert vector[0] == pytest.approx(1.0)


def test_a_vector_instance_still_answers_a_position():
    """Naming a component must not replace the positional subscript."""
    vector = float3(1.0, 2.0, 3.0)
    assert vector[1] == pytest.approx(2.0)
    vector["y"] = 9.0
    assert vector[1] == pytest.approx(9.0)


def test_a_structure_instance_answers_a_member_name():
    ray = Ray(point3f(1.0, 2.0, 3.0), float3(4.0, 5.0, 6.0))
    assert ray["origin"]["y"] == pytest.approx(2.0)
    assert ray["direction"]["z"] == pytest.approx(6.0)
