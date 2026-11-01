"""Structures, unions and enums a caller declares, in the two ctypes dialects and numpy.

Three layouts reach a device besides a hand-written record dtype, and each has a shape the
translator has to keep:

* `ctypes.LittleEndianStructure` is a `ctypes.Structure` subclass, so it needs nothing new --
  but its *members* are little-endian spellings, and a compiler on a big-endian host would
  lay them out differently. It is pinned here so the four backends keep answering the same;
* `ctypes.Union` / `ctypes.LittleEndianUnion` are **siblings** of `ctypes.Structure` in
  `ctypes`, not subclasses, so every "is this a structure?" test has to ask for both. Their
  members share one address, which numpy records as a list of zero offsets -- and that is
  what makes a hand-written `np.dtype` of the same shape the same device type;
* `enum.Enum` / `enum.IntEnum` members are named constants: the device reads the value, and
  the enum class itself never reaches the generated code.
"""
from __future__ import annotations

import ctypes
import enum

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# The declarations used below
# ---------------------------------------------------------------------------


class LittleEndianPair(ctypes.LittleEndianStructure):
    _fields_ = [("i", ctypes.c_int32), ("f", ctypes.c_float)]


class Union(ctypes.Union):
    _fields_ = [("i", ctypes.c_int32), ("f", ctypes.c_float)]


class LittleEndianUnion(ctypes.LittleEndianUnion):
    _fields_ = [("i", ctypes.c_int32), ("f", ctypes.c_float)]


class Nested(ctypes.Structure):
    """A structure holding a union, which is how a tagged value is spelled."""

    _fields_ = [("tag", ctypes.c_int32), ("value", Union)]


OVERLAPPING = np.dtype({"names": ["i", "f"], "formats": ["<i4", "<f4"],
                        "offsets": [0, 0], "itemsize": 4})


class Kind(enum.IntEnum):
    FIRST = 1
    SECOND = 2


class Colour(enum.Enum):
    RED = 1
    GREEN = 2


# ---------------------------------------------------------------------------
# ctypes.LittleEndianStructure
# ---------------------------------------------------------------------------


@kernel
def read_little_endian(p, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p["i"] + p["f"]


def test_a_little_endian_structure_reads_like_a_structure(runner):
    """It is a `ctypes.Structure` subclass, so its members travel as members."""
    out = np.zeros(1, dtype=np.float32)
    runner(read_little_endian, LittleEndianPair(3, 0.5), out)
    assert float(out[0]) == pytest.approx(3.5)


def test_a_little_endian_structure_is_a_declared_structure():
    """The dtype carries the same evidence a native structure's does."""
    from pygpgpu.kernel import structs

    dtype = np.dtype(LittleEndianPair)
    assert structs.is_declared_structure(dtype)
    assert structs.is_ctypes_record(LittleEndianPair)
    assert not structs.is_union_dtype(dtype)


# ---------------------------------------------------------------------------
# ctypes.Union and ctypes.LittleEndianUnion
# ---------------------------------------------------------------------------


@kernel
def read_union_i(p, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p["i"]


@kernel
def read_union_f(p, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p["f"]


@kernel
def read_union_attribute(p, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p.i


def test_a_union_is_a_ctypes_record():
    """A union is not a `Structure` subclass, so it is recognized on its own."""
    from pygpgpu.kernel import structs

    assert structs.is_ctypes_record(Union())
    assert structs.is_ctypes_record(LittleEndianUnion())
    assert not issubclass(Union, ctypes.Structure)
    dtype = structs.structure_dtype_of(Union())
    assert dtype is not None
    assert structs.is_union_dtype(dtype)
    # ... and the same record written by hand is that same shape.
    assert structs.is_union_dtype(OVERLAPPING)
    assert structs.is_declared_structure(OVERLAPPING)


@pytest.mark.parametrize("declaration, value, read, expected", [
    (Union, 5, read_union_i, 5.0),
    (LittleEndianUnion, 5, read_union_i, 5.0),
    (Union, 1.5, read_union_f, 1.5),
    (Union, 9, read_union_attribute, 9.0),
])
def test_union_members_are_read_on_every_backend(runner, declaration, value, read, expected):
    """Both union classes answer both member spellings, on every backend."""
    instance = declaration()
    member = "f" if read is read_union_f else "i"
    setattr(instance, member, value)
    out = np.zeros(1, dtype=np.float32)
    runner(read, instance, out)
    assert float(out[0]) == pytest.approx(expected)


def test_a_union_member_write_reaches_the_caller(runner):
    """A union travels as a pointer, so a member store lands in the caller's value."""

    @kernel
    def writes_union(p, out: np.ndarray):
        p.i = 7
        for i in range(out.shape[0]):
            out[i] = p.i

    instance = Union()
    instance.i = 0
    out = np.zeros(1, dtype=np.float32)
    runner(writes_union, instance, out)
    assert float(out[0]) == pytest.approx(7.0)
    assert instance.i == 7


@kernel
def read_nested_union(p, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = p["tag"] + p["value"]["i"]


def test_a_union_member_of_a_structure(runner):
    """The union's declaration is emitted before the structure that holds it."""
    nested = Nested(tag=2)
    nested.value.i = 3
    out = np.zeros(1, dtype=np.float32)
    runner(read_nested_union, nested, out)
    assert float(out[0]) == pytest.approx(5.0)


@kernel
def read_overlapping_rows(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[0]["i"]


def test_an_overlapping_np_dtype_is_a_union(runner):
    """A hand-written dtype whose fields all sit at offset zero is the same layout."""
    rows = np.zeros(2, dtype=OVERLAPPING)
    rows["i"][0] = 7
    out = np.zeros(1, dtype=np.float32)
    runner(read_overlapping_rows, rows, out)
    assert float(out[0]) == pytest.approx(7.0)


@kernel
def read_overlapping_record(record: np.void, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = record["i"]


def test_an_overlapping_record_by_value(runner):
    """`bytes(record)` is refused for an overlapping dtype, so the launch copies the raw
    bytes instead -- which is what makes a by-value union record uploadable."""
    record = np.zeros(1, dtype=OVERLAPPING)[0]
    record["i"] = 7
    out = np.zeros(1, dtype=np.float32)
    runner(read_overlapping_record, record, out)
    assert float(out[0]) == pytest.approx(7.0)


def test_a_union_is_declared_as_a_union(gpu_backend):
    """The generated source says `union`, not `struct`: `struct` would double its width."""
    import glob
    import pathlib
    import shutil

    @kernel
    def takes_union(p, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = p["i"]

    cache_dir = takes_union._cache_dir(gpu_backend)
    shutil.rmtree(cache_dir, ignore_errors=True)
    takes_union[gpu_backend](Union(), np.zeros(1, dtype=np.float32))
    extension = {"opencl": "cl", "cuda": "cu", "hip": "hip"}[gpu_backend]
    files = sorted(glob.glob(f"{cache_dir}/*.{extension}"))
    assert files, f"no generated source under {cache_dir}"
    source = "\n".join(pathlib.Path(path).read_text(encoding="utf-8") for path in files)
    # `typedef union {` on OpenCL, `union U_<hash>` on CUDA/HIP -- the keyword leads.
    assert "union" in source, source
    assert "_Union_" in source, source


# ---------------------------------------------------------------------------
# enum.Enum / enum.IntEnum
# ---------------------------------------------------------------------------


@kernel
def reads_int_enum(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float(Kind.SECOND)


@kernel
def reads_plain_enum(out: np.ndarray):
    """A plain `Enum` member used where a number is wanted is its value."""
    for i in range(out.shape[0]):
        out[i] = float(Colour.GREEN)


@kernel
def reads_int_enum_from_a_local(out: np.ndarray):
    member = Kind.FIRST
    for i in range(out.shape[0]):
        out[i] = float(member)


def test_an_int_enum_member_is_its_value(runner):
    """`Kind.SECOND` means the number, on every backend."""
    out = np.zeros(1, dtype=np.float32)
    runner(reads_int_enum, out)
    assert float(out[0]) == pytest.approx(2.0)


def test_a_plain_enum_member_is_its_value(gpu_backend):
    """`Colour.GREEN` is not an int, so the translator has to supply the value.

    The interpreter backend runs the body unchanged, and `float(Colour.GREEN)` is a
    `TypeError` in python itself -- so this spelling only has a device answer, and the
    interpreter keeps answering what python does.
    """
    out = np.zeros(1, dtype=np.float32)
    reads_plain_enum[gpu_backend](out)
    assert float(out[0]) == pytest.approx(2.0)


def test_an_enum_member_through_a_local(runner):
    """The member a body binds to a local name is the same constant."""
    out = np.zeros(1, dtype=np.float32)
    runner(reads_int_enum_from_a_local, out)
    assert float(out[0]) == pytest.approx(1.0)


def test_an_enum_member_reaches_a_generated_source_as_its_value(gpu_backend):
    """Nothing in the generated source names the enum: only its number appears."""
    import glob
    import pathlib
    import shutil

    @kernel
    def takes_kind(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = float(Kind.SECOND)

    cache_dir = takes_kind._cache_dir(gpu_backend)
    shutil.rmtree(cache_dir, ignore_errors=True)
    takes_kind[gpu_backend](np.zeros(1, dtype=np.float32))
    extension = {"opencl": "cl", "cuda": "cu", "hip": "hip"}[gpu_backend]
    files = sorted(glob.glob(f"{cache_dir}/*.{extension}"))
    assert files, f"no generated source under {cache_dir}"
    source = "\n".join(pathlib.Path(path).read_text(encoding="utf-8") for path in files)
    assert "enum Kind" not in source, source
    assert "2" in source, source


def test_an_enum_member_with_a_non_numeric_value_is_refused(gpu_backend):
    """Only a number has a device meaning, and the message says which member it was."""
    if gpu_backend == "python":
        pytest.skip("the interpreter runs the body unchanged")

    class Named(enum.Enum):
        A = "text"

    @kernel
    def takes_named(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = float(len(Named.A.value))

    with pytest.raises(SyntaxError) as caught:
        takes_named[gpu_backend](np.zeros(1, dtype=np.float32))
    assert "Named.A" in str(caught.value), str(caught.value)
