"""A class's magic methods become device functions and are called where they trigger.

A python class can give an operator a meaning of its own: `a + b` on a class defining
`__add__` is that method, and a kernel body that writes `a + b` must keep it. The
translation turns each such operator into a device call to the method -- the same shape a
plain method call takes (`ray.at(t)` becomes `Ray_at`) -- so the method is generated once,
as its own device function, and called where the operator appears.

Covered here: binary operators, the reflected spelling (`2.0 * v` reaching `__rmul__`),
comparisons (including the reversed reading python applies to `a < b`), and the unary
operators, plus `__getitem__` and `__len__`.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from pygpgpu import kernel


class Vec(ctypes.Structure):
    """A three-component value whose arithmetic is written out in python."""

    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, factor):
        return Vec(self.x * factor, self.y * factor, self.z * factor)

    def __neg__(self):
        return Vec(-self.x, -self.y, -self.z)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.x < other.x


class Box(ctypes.Structure):
    """A one-value class that only knows the *reflected* forms."""

    _fields_ = [("value", ctypes.c_float)]

    def __radd__(self, other):
        return Box(other + self.value)

    def __rmul__(self, other):
        return Box(other * self.value)

    def __getitem__(self, index):
        return self.value

    def __len__(self):
        return 4


@kernel
def adds(a, b, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (a + b).x


@kernel
def subtracts(a, b, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (a - b).y


@kernel
def scales(a, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (a * 2.0).z


@kernel
def negates(a, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (-a).x


@kernel
def compares_equal(a, b, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a == b else 0.0


@kernel
def compares_less(a, b, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a < b else 0.0


@kernel
def reflected_add(box, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (2.0 + box).value


@kernel
def reflected_mul(box, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (3.0 * box).value


@kernel
def subscripts(box, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = box[0]


@kernel
def measures_length(box, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = len(box)


def test_a_binary_operator_becomes_a_device_call(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(adds, Vec(1.0, 2.0, 3.0), Vec(4.0, 5.0, 6.0), out)
    assert float(out[0]) == pytest.approx(5.0)


def test_every_binary_operator_keeps_its_meaning(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(subtracts, Vec(1.0, 9.0, 3.0), Vec(4.0, 4.0, 6.0), out)
    assert float(out[0]) == pytest.approx(5.0)
    runner(scales, Vec(1.0, 2.0, 3.0), out)
    assert float(out[0]) == pytest.approx(6.0)


def test_a_unary_operator_becomes_a_device_call(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(negates, Vec(1.0, 2.0, 3.0), out)
    assert float(out[0]) == pytest.approx(-1.0)


def test_a_comparison_becomes_a_device_call(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(compares_equal, Vec(1.0, 2.0, 0.0), Vec(1.0, 2.0, 0.0), out)
    assert float(out[0]) == pytest.approx(1.0)
    runner(compares_equal, Vec(1.0, 2.0, 0.0), Vec(1.0, 3.0, 0.0), out)
    assert float(out[0]) == pytest.approx(0.0)
    runner(compares_less, Vec(1.0, 0.0, 0.0), Vec(2.0, 0.0, 0.0), out)
    assert float(out[0]) == pytest.approx(1.0)


def test_the_reflected_spelling_is_kept(runner):
    """`2.0 + box` reaches `box.__radd__`, as python resolves it."""
    out = np.zeros(1, dtype=np.float32)
    runner(reflected_add, Box(3.0), out)
    assert float(out[0]) == pytest.approx(5.0)
    runner(reflected_mul, Box(4.0), out)
    assert float(out[0]) == pytest.approx(12.0)


def test_a_subscript_and_a_length_use_the_class_methods(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(subscripts, Box(7.0), out)
    assert float(out[0]) == pytest.approx(7.0)
    runner(measures_length, Box(1.0), out)
    assert float(out[0]) == pytest.approx(4.0)


@kernel
def generated_once(a, b, out: np.ndarray):
    """A kernel of its own, so this test's launch always writes the module's source."""
    for i in range(out.shape[0]):
        out[i] = (a + b).x


def test_the_magic_method_is_one_device_function(gpu_backend):
    """The method is generated once, named `Class___method__`, and called, not inlined."""
    import glob
    import pathlib
    import shutil

    cache_dir = generated_once._cache_dir(gpu_backend)
    shutil.rmtree(cache_dir, ignore_errors=True)
    generated_once[gpu_backend](Vec(1.0, 2.0, 3.0), Vec(4.0, 5.0, 6.0),
                                np.zeros(1, dtype=np.float32))
    extension = {"opencl": "cl", "cuda": "cu", "hip": "hip"}[gpu_backend]
    files = sorted(glob.glob(f"{cache_dir}/*.{extension}"))
    assert files, f"no generated source under {cache_dir}"
    # One file per python module holds every kernel it defines, so the whole module's
    # source is what is checked.
    source = "\n".join(pathlib.Path(path).read_text(encoding="utf-8")
                       for path in files)
    # Vec___add___<hash>(...): the method as its own device function, named after the
    # class it belongs to, and called from the kernel rather than inlined there.
    assert "Vec___add___" in source, source
