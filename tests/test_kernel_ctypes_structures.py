"""Structures declared as `ctypes.Structure` classes, alongside record dtypes.

`np.dtype(cls)` over a `ctypes.Structure` produces the same kind of record a hand-written
`np.dtype([("i", "<i4"), ...])` does, so both forms are two spellings of one mechanism. A
class additionally carries something a bare dtype cannot: the *declaration*. `{float x;
float y;}` has the layout of `float2` down to the byte, and only the dtype object's
alignment flag says which one the caller meant -- a ctypes class always yields an aligned
record, while the vector dtypes this package ships are not aligned.

These tests pin both directions: a class is a structure even when it looks like a vector,
and a vector is still a vector.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest
from pygf import float3

from pygpgpu import kernel, vec_types
from pygpgpu.kernel import structs
from pygpgpu.kernel.translator import infer_ctype


class Pair(ctypes.Structure):
    _fields_ = [("i", ctypes.c_int), ("f", ctypes.c_float)]


class Inner(ctypes.Structure):
    _fields_ = [("a", ctypes.c_int), ("b", ctypes.c_int)]


class Nested(ctypes.Structure):
    _fields_ = [("inner", Inner), ("c", ctypes.c_float)]


class WithVector(ctypes.Structure):
    _fields_ = [("v", float3), ("k", ctypes.c_int)]


class WithArray(ctypes.Structure):
    _fields_ = [("n", ctypes.c_int), ("xs", ctypes.c_float * 3)]


class Vec2Like(ctypes.Structure):
    """The layout of `float2`, declared as a structure."""

    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


# ---------------------------------------------------------------------------
# The type a value denotes
# ---------------------------------------------------------------------------


def test_a_ctypes_class_is_a_structure_value():
    assert infer_ctype(Pair(3, 0.5)) == structs.register(Pair)


def test_a_ctypes_class_argument_is_registered_by_its_class():
    """Passing the class itself (not an instance) names the same device type."""
    assert infer_ctype(Pair) == structs.register(Pair)


def test_a_ctypes_array_is_a_structure_array():
    assert infer_ctype(np.zeros(2, dtype=Pair)) == ("*", structs.register(Pair))


def test_a_structure_that_looks_like_a_vector_stays_a_structure():
    """`Vec2Like` and `float2` have byte-identical layouts; the declaration decides."""
    assert infer_ctype(np.zeros(1, dtype=Vec2Like)) == ("*", structs.register(Vec2Like))
    assert infer_ctype(Vec2Like(1.0, 2.0)) == structs.register(Vec2Like)


def test_a_vector_is_still_a_vector():
    assert infer_ctype(vec_types.float2(1.0, 2.0)) == "float2"
    assert infer_ctype(np.zeros(1, dtype=vec_types.float2.dtype)) == ("*", "float2")
    assert infer_ctype(float3(1, 2, 3)) == "float3"


def test_a_vector_alias_is_still_a_vector():
    from pygf import color3f

    assert infer_ctype(color3f(1.0, 2.0, 3.0)) == "float3"


def test_the_alignment_flag_is_what_separates_them():
    declared = np.zeros(1, dtype=Vec2Like).dtype
    vector = np.zeros(1, dtype=vec_types.float2.dtype).dtype
    assert declared == vector          # same layout
    assert declared.isalignedstruct    # ... but a C structure's alignment
    assert not vector.isalignedstruct
    assert structs.is_declared_structure(declared)
    assert not structs.is_declared_structure(vector)


def test_a_plain_record_dtype_is_a_structure():
    dtype = np.dtype([("i", "<i4"), ("f", "<f4")])
    assert structs.is_record(dtype)
    assert infer_ctype(np.zeros(2, dtype=dtype)) == ("*", structs.register(dtype))


# ---------------------------------------------------------------------------
# Through a launch
# ---------------------------------------------------------------------------


@kernel
def read_pair_i(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[i]["i"]


@kernel
def read_nested(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[i]["inner"]["a"] + rows[i]["c"]


@kernel
def read_vector_member(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[i]["v"]["x"] + rows[i]["k"]


@kernel
def read_array_member(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[i]["xs"][1]


@kernel
def read_vec2like(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[i]["y"]


@kernel
def take_pair(p, out: np.ndarray):
    """A structure by value: the instance travels as the record it describes."""
    for i in range(out.shape[0]):
        out[i] = p["i"]


def test_a_ctypes_array_reads(runner):
    rows = np.zeros(2, dtype=Pair)
    rows["i"] = [3, 9]
    out = np.zeros(1, dtype=np.float32)
    runner(read_pair_i, rows, out)
    assert float(out[0]) == 3.0


def test_nested_ctypes_structures(runner):
    rows = np.zeros(1, dtype=Nested)
    rows["inner"]["a"] = 2
    rows["c"] = 0.5
    out = np.zeros(1, dtype=np.float32)
    runner(read_nested, rows, out)
    assert float(out[0]) == pytest.approx(2.5)


def test_a_vector_member_of_a_ctypes_structure(runner):
    rows = np.zeros(1, dtype=WithVector)
    rows["v"] = (1.0, 2.0, 3.0)
    rows["k"] = 4
    out = np.zeros(1, dtype=np.float32)
    runner(read_vector_member, rows, out)
    assert float(out[0]) == pytest.approx(5.0)


def test_an_array_member_of_a_ctypes_structure(runner):
    rows = np.zeros(1, dtype=WithArray)
    rows["xs"][0][1] = 7.0
    out = np.zeros(1, dtype=np.float32)
    runner(read_array_member, rows, out)
    assert float(out[0]) == pytest.approx(7.0)


def test_a_vector_shaped_ctypes_structure_is_read_by_its_field(runner):
    """The member is `["y"]`, a structure's spelling, not a vector's `.y`."""
    rows = np.zeros(1, dtype=Vec2Like)
    rows["y"] = 5.0
    out = np.zeros(1, dtype=np.float32)
    runner(read_vec2like, rows, out)
    assert float(out[0]) == pytest.approx(5.0)


def test_a_ctypes_structure_by_value(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(take_pair, Pair(3, 0.5), out)
    assert float(out[0]) == 3.0


@kernel
def create_ctypes_array(out: np.ndarray):
    made = np.zeros(2, dtype=Pair)
    made[1]["i"] = 8
    for i in range(out.shape[0]):
        out[i] = made[1]["i"]
    return made


def test_a_ctypes_array_can_be_created_in_the_body(runner):
    out = np.zeros(1, dtype=np.float32)
    created = runner(create_ctypes_array, out)
    assert float(out[0]) == pytest.approx(8.0)
    assert created["i"][1] == 8


@kernel
def return_element(rows: np.ndarray):
    for i in range(rows.shape[0]):
        return rows[i]


def test_a_ctypes_element_can_be_returned(runner):
    rows = np.zeros(1, dtype=Pair)
    rows["i"] = 4
    returned = runner(return_element, rows)
    assert returned["i"] == 4
