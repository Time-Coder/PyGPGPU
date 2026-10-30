"""Structures as kernel values: parameters, returns, created values, outer scope.

A structure is a numpy record on the host: `np.dtype([("i", "<i4"), ("f", "<f4")])`. That
describes the layout completely, so it is also what the device type is built from -- no
`ctypes` class is needed, and a record array is simply an array of that type.

Member access keeps numpy's spelling: `s["i"]` reads the field, and `s["v"].x` reads a vector
member's component. A structure may hold scalars, vectors, 1/2/3-D arrays, other structures,
and arrays of structures; every one of those combinations is pinned here, for all four places
a value can cross the boundary: an argument, a return, an object the body created, and a
variable from the enclosing scope.
"""
from __future__ import annotations

import numpy as np
import pytest
from pygf import float3

from pygpgpu import kernel

# ---------------------------------------------------------------------------
# The structures used below
# ---------------------------------------------------------------------------

PAIR = np.dtype([("i", "<i4"), ("f", "<f4")])
WITH_VECTOR = np.dtype([("v", float3.numpy_dtype()), ("k", "<i4")])
WITH_ARRAY = np.dtype([("n", "<i4"), ("xs", "<f4", (3,))])
INNER = np.dtype([("a", "<i4"), ("b", "<i4")])
NESTED = np.dtype([("inner", INNER), ("c", "<f4")])
WITH_STRUCT_ARRAY = np.dtype([("tag", "<i4"), ("rows", PAIR, (2,))])


def pair(i: int, f: float) -> np.void:
    record = np.zeros(1, dtype=PAIR)[0]
    record["i"] = i
    record["f"] = f
    return record


# ---------------------------------------------------------------------------
# 1. A structure as an argument
# ---------------------------------------------------------------------------


@kernel
def read_pair(p: np.void, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p["i"] + p["f"]


@kernel
def read_pair_vector(p: np.void, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p["v"]["x"] + p["v"]["z"] + p["k"]


@kernel
def read_pair_array_member(p: np.void, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p["xs"][1]


@kernel
def read_nested(p: np.void, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p["inner"]["a"] + p["c"]


@kernel
def read_struct_array_member(p: np.void, out: np.ndarray):
    """A structure whose member is an array of structures."""
    for i in range(out.shape[0]):
        out[i] = p["rows"][1]["i"] + p["tag"]


def test_a_structure_argument_holding_an_array_of_structures(runner):
    record = np.zeros(1, dtype=WITH_STRUCT_ARRAY)[0]
    record["tag"] = 5
    record["rows"][1]["i"] = 3
    out = np.zeros(1, dtype=np.float32)
    runner(read_struct_array_member, record, out)
    assert float(out[0]) == pytest.approx(8.0)


def test_a_structure_argument(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(read_pair, pair(3, 0.5), out)
    assert float(out[0]) == pytest.approx(3.5)


def test_a_structure_argument_holding_a_vector(runner):
    record = np.zeros(1, dtype=WITH_VECTOR)[0]
    record["v"] = (1.0, 2.0, 3.0)
    record["k"] = 4
    out = np.zeros(1, dtype=np.float32)
    runner(read_pair_vector, record, out)
    assert float(out[0]) == pytest.approx(8.0)


def test_a_structure_argument_holding_an_array(runner):
    record = np.zeros(1, dtype=WITH_ARRAY)[0]
    record["n"] = 2
    record["xs"][1] = 7.0
    out = np.zeros(1, dtype=np.float32)
    runner(read_pair_array_member, record, out)
    assert float(out[0]) == pytest.approx(7.0)


def test_a_structure_argument_holding_a_nested_structure(runner):
    record = np.zeros(1, dtype=NESTED)[0]
    record["inner"]["a"] = 2
    record["c"] = 0.5
    out = np.zeros(1, dtype=np.float32)
    runner(read_nested, record, out)
    assert float(out[0]) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# 2. An array of structures, one to three dimensions
# ---------------------------------------------------------------------------


@kernel
def read_struct_array1(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[i]["i"]


@kernel
def read_struct_array2(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[i, 0]["i"]


@kernel
def read_struct_array3(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[i, 0, 0]["i"]


@kernel
def read_struct_array_with_vector(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[i]["v"]["y"]


@kernel
def write_struct_array1(rows: np.ndarray):
    for i in range(rows.shape[0]):
        rows[i]["i"] = i + 1
    return rows


@kernel
def write_struct_array2(rows: np.ndarray):
    for i, j in np.ndindex(rows.shape[0], rows.shape[1]):
        rows[i, j]["i"] = i * 2 + j + 1
    return rows


@kernel
def write_struct_array3(rows: np.ndarray):
    for i, j, k in np.ndindex(rows.shape[0], rows.shape[1], rows.shape[2]):
        rows[i, j, k]["i"] = i * 4 + j * 2 + k + 1
    return rows


@pytest.mark.parametrize("kernel_object, shape", [
    (write_struct_array1, (4,)),
    (write_struct_array2, (2, 2)),
    (write_struct_array3, (2, 2, 2)),
])
def test_an_array_of_structures_round_trips(runner, kernel_object, shape):
    """A structure array written by the kernel reaches the caller's own array."""
    rows = np.zeros(shape, dtype=PAIR)
    runner(kernel_object, rows)
    np.testing.assert_array_equal(rows["i"], np.arange(1, rows.size + 1).reshape(shape))


def test_a_one_dimensional_structure_array_is_readable(runner):
    rows = np.zeros(2, dtype=PAIR)
    rows["i"] = [3, 7]
    out = np.zeros(1, dtype=np.float32)
    runner(read_struct_array1, rows, out)
    assert float(out[0]) == 3.0


def test_a_two_dimensional_structure_array_is_readable(runner):
    rows = np.zeros((2, 2), dtype=PAIR)
    rows[0, 0]["i"] = 5
    out = np.zeros(1, dtype=np.float32)
    runner(read_struct_array2, rows, out)
    assert float(out[0]) == 5.0


def test_a_three_dimensional_structure_array_is_readable(runner):
    rows = np.zeros((2, 2, 2), dtype=PAIR)
    rows[0, 0, 0]["i"] = 9
    out = np.zeros(1, dtype=np.float32)
    runner(read_struct_array3, rows, out)
    assert float(out[0]) == 9.0


def test_a_structure_array_holding_vectors(runner):
    rows = np.zeros(2, dtype=WITH_VECTOR)
    rows[0]["v"] = (1.0, 2.0, 3.0)
    out = np.zeros(1, dtype=np.float32)
    runner(read_struct_array_with_vector, rows, out)
    assert float(out[0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 3. A structure as a return value
# ---------------------------------------------------------------------------


@kernel
def return_pair(p: np.void, x: np.ndarray):
    """A structure value handed back: the slot is a one-element record buffer."""
    for i in range(x.shape[0]):
        return p


def test_a_structure_can_be_returned(runner):
    returned = runner(return_pair, pair(4, 0.5), np.zeros(1, dtype=np.float32))
    assert returned["i"] == 4
    assert returned["f"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. A structure created inside the body, and one from the enclosing scope
# ---------------------------------------------------------------------------

OUTER_PAIR: np.void = pair(6, 0.25)
OUTER_STRUCT_ARRAY: np.ndarray = np.zeros(2, dtype=PAIR)


@kernel
def read_outer_struct(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = OUTER_PAIR["i"]


@kernel
def read_outer_struct_array(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = OUTER_STRUCT_ARRAY[1]["i"]


@kernel
def create_struct_array(out: np.ndarray):
    made: np.ndarray = np.zeros(2, dtype=np.dtype([("i", "<i4"), ("f", "<f4")]))
    made[1]["i"] = 8
    for i in range(out.shape[0]):
        out[i] = made[1]["i"]
    return made


def test_an_outer_structure_is_visible(gpu_backend, runner):
    out = np.zeros(1, dtype=np.float32)
    runner(read_outer_struct, out)
    assert float(out[0]) == 6.0


def test_an_outer_structure_array_is_visible(gpu_backend, runner):
    OUTER_STRUCT_ARRAY[1]["i"] = 7
    out = np.zeros(1, dtype=np.float32)
    runner(read_outer_struct_array, out)
    assert float(out[0]) == 7.0


def test_a_structure_array_can_be_created_in_the_body(gpu_backend, runner):
    out = np.zeros(1, dtype=np.float32)
    created = runner(create_struct_array, out)
    assert float(out[0]) == 8.0
    assert created["i"][1] == 8
