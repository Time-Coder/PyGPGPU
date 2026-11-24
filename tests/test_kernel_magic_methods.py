"""A class's magic methods become device functions and are called where they trigger.

A python class can give an operator a meaning of its own: `a + b` on a class defining
`__add__` is that method, and a kernel body that writes `a + b` must keep it. The
translation turns each such operator into a device call to the method -- the same shape a
plain method call takes (`ray.at(t)` becomes `Ray_at`) -- so the method is generated once,
as its own device function, and called where the operator appears.

Covered here: binary operators, the reflected spelling (`2.0 * v` reaching `__rmul__`),
comparisons (including the reversed reading python applies to `a < b`), and the unary
operators, plus `__getitem__` (on a name and on a receiver member), `__setitem__`,
`__len__`, `__bool__` (in `if`, `not`, a conditional expression and an `and` chain, with
`__len__` as python's fallback), the refusal of iteration -- python's `__iter__` has no
device meaning -- and a `@kernel`-decorated operator, which an operator can only use as a
device helper.
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


# ---------------------------------------------------------------------------
# Truth: `if box:` asks the class, not the pointer
# ---------------------------------------------------------------------------


class Truth(ctypes.Structure):
    """Five floats -- no vector type has this width, so the layout stays a structure.

    `__bool__` is the meaning python gives `if box:`, and testing the structure pointer
    instead answers "true" for every argument, which is the wrong branch with no error.
    """

    _fields_ = [(name, ctypes.c_float) for name in ("a", "b", "c", "d", "e")]

    def __bool__(self):
        return self.a != 0.0


class Sized(ctypes.Structure):
    """Six floats, and only `__len__`: python's truth falls back to `len(x) != 0`."""

    _fields_ = [(name, ctypes.c_float) for name in ("a", "b", "c", "d", "e", "f")]

    def __len__(self):
        return 3


class NoTruth(ctypes.Structure):
    """Eight floats with neither method: there is no truth the device could compute."""

    _fields_ = [(name, ctypes.c_float)
                for name in ("a", "b", "c", "d", "e", "f", "g", "h")]


@kernel
def takes_the_branch(box, out: np.ndarray):
    for i in range(out.shape[0]):
        if box:
            out[i] = 1.0
        else:
            out[i] = 0.0


@kernel
def negates_the_truth(box, out: np.ndarray):
    for i in range(out.shape[0]):
        if not box:
            out[i] = 1.0
        else:
            out[i] = 0.0


@kernel
def chooses_by_truth(box, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 5.0 if box else 7.0


@kernel
def conjunctive_truth(box, other, out: np.ndarray):
    for i in range(out.shape[0]):
        if box and other:
            out[i] = 1.0
        else:
            out[i] = 0.0


@kernel
def sized_branch(box, out: np.ndarray):
    for i in range(out.shape[0]):
        if box:
            out[i] = 1.0
        else:
            out[i] = 0.0


@kernel
def brancheless_truth(box, out: np.ndarray):
    for i in range(out.shape[0]):
        if box:
            out[i] = 1.0
        else:
            out[i] = 0.0


def test_a_condition_asks_the_class_bool(runner):
    """`if box:` is `box.__bool__()`, for a false and a true value."""
    out = np.zeros(1, dtype=np.float32)
    runner(takes_the_branch, Truth(0.0, 1.0, 2.0, 3.0, 4.0), out)
    assert float(out[0]) == pytest.approx(0.0)
    runner(takes_the_branch, Truth(1.0, 0.0, 0.0, 0.0, 0.0), out)
    assert float(out[0]) == pytest.approx(1.0)


def test_not_and_the_conditional_expression_ask_it_too(runner):
    """`not box` and `a if box else b` are the same truth in two other positions."""
    out = np.zeros(1, dtype=np.float32)
    runner(negates_the_truth, Truth(0.0, 0.0, 0.0, 0.0, 0.0), out)
    assert float(out[0]) == pytest.approx(1.0)
    runner(chooses_by_truth, Truth(0.0, 0.0, 0.0, 0.0, 0.0), out)
    assert float(out[0]) == pytest.approx(7.0)
    runner(chooses_by_truth, Truth(2.0, 0.0, 0.0, 0.0, 0.0), out)
    assert float(out[0]) == pytest.approx(5.0)


def test_an_and_chain_reads_each_operands_truth(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(conjunctive_truth, Truth(0.0, 0.0, 0.0, 0.0, 0.0),
           Truth(1.0, 0.0, 0.0, 0.0, 0.0), out)
    assert float(out[0]) == pytest.approx(0.0)
    runner(conjunctive_truth, Truth(1.0, 0.0, 0.0, 0.0, 0.0),
           Truth(1.0, 0.0, 0.0, 0.0, 0.0), out)
    assert float(out[0]) == pytest.approx(1.0)


def test_len_is_the_fallback_truth_like_in_python(runner):
    """A class with `__len__` only: python answers `len(x) != 0`."""
    out = np.zeros(1, dtype=np.float32)
    runner(sized_branch, Sized(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), out)
    assert float(out[0]) == pytest.approx(1.0)


def test_a_class_without_a_truth_value_is_refused(gpu_backend):
    """No `__bool__` and no `__len__` means no truth a device can compute.

    The refusal matters: the pointer test this used to emit answered "true" for every
    argument, so `if box:` silently took the first branch forever.
    """
    with pytest.raises(SyntaxError) as caught:
        brancheless_truth[gpu_backend](NoTruth(*([1.0] * 8)),
                                       np.zeros(1, dtype=np.float32))
    assert "__bool__" in str(caught.value)


# ---------------------------------------------------------------------------
# Subscripts: `__getitem__` on a member, `__setitem__` on a store
# ---------------------------------------------------------------------------


class Slot(ctypes.Structure):
    """Seven floats; a class that answers its own subscripts both ways."""

    _fields_ = [(name, ctypes.c_float)
                for name in ("a", "b", "c", "d", "e", "f", "g")]

    def __getitem__(self, index):
        return self.a + index * 10.0

    def __setitem__(self, index, value):
        self.a = value + index


@kernel
def stores_through_setitem(box, out: np.ndarray):
    for i in range(out.shape[0]):
        box[2] = 10.0
        out[i] = box[3]


class SlotHolder:
    """A kernel whose receiver carries a class-typed member."""

    def __init__(self):
        self.slot = Slot(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @kernel
    def reads_member_by_index(self, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = self.slot[1]

    @kernel
    def writes_member_by_index(self, value, out: np.ndarray):
        for i in range(out.shape[0]):
            self.slot[0] = value
            out[i] = self.slot.a


def test_a_store_through_setitem_reaches_the_class(runner):
    """`box[i] = v` is `box.__setitem__(i, v)`, and the store reaches the caller."""
    out = np.zeros(1, dtype=np.float32)
    box = Slot(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    runner(stores_through_setitem, box, out)
    assert float(out[0]) == pytest.approx(42.0)      # a = 12, then 12 + 30
    assert float(box.a) == pytest.approx(12.0)


def test_a_member_that_holds_a_class_is_indexed_by_its_method(runner):
    """`self.slot[i]` and `self.slot[i] = v` are the class's methods, not a buffer index."""
    holder = SlotHolder()
    out = np.zeros(1, dtype=np.float32)
    runner(holder.reads_member_by_index, out)
    assert float(out[0]) == pytest.approx(12.0)      # 2 + 10
    runner(holder.writes_member_by_index, 5.0, out)
    assert float(out[0]) == pytest.approx(5.0)
    assert float(holder.slot.a) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Iteration: `__iter__` has no device meaning
# ---------------------------------------------------------------------------


@kernel
def iterates_a_class(box, out: np.ndarray):
    total = 0.0
    for value in box:
        total = total + value
    for i in range(out.shape[0]):
        out[i] = total


def test_iterating_a_class_value_is_refused(gpu_backend):
    """`for x in box` on a class value has no device translation.

    The loop used to read the structure pointer as a buffer of records -- and above the
    dispatch loop it disappeared entirely, so the kernel answered a stale value with
    nothing reported. Refusing says which method python would have used.
    """
    with pytest.raises(SyntaxError) as caught:
        iterates_a_class[gpu_backend](Truth(1.0, 2.0, 3.0, 4.0, 5.0),
                                      np.zeros(1, dtype=np.float32))
    assert "__iter__" in str(caught.value)


# ---------------------------------------------------------------------------
# A `@kernel` magic method is a kernel on the host and a helper for the operator
# ---------------------------------------------------------------------------


class KernelVec(ctypes.Structure):
    """Thirteen floats -- its own layout -- and an operator decorated as a kernel."""

    _fields_ = [(name, ctypes.c_float)
                for name in ("a", "b", "c", "d", "e", "f", "g", "h", "i",
                             "j", "k", "l", "m")]

    @kernel
    def __add__(self, other):
        return KernelVec(self.a + other.a, self.b, self.c, self.d, self.e, self.f,
                         self.g, self.h, self.i, self.j, self.k, self.l, self.m)


@kernel
def adds_kernel_magic(x, y, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = (x + y).a


def test_a_kernel_decorated_operator_is_a_device_helper(gpu_backend):
    """An operator cannot carry a work size, so the decorated body is translated as a helper.

    The decoration stays useful on the host (the method is still a launchable kernel), which
    is why `_magic_method_of` unwraps `PyKernel.raw_func` instead of refusing. The interpreter
    backend runs the body as written, where `x + y` *is* a call to the decorated method -- a
    kernel launch -- so this shape belongs to the device backends.
    """
    out = np.zeros(1, dtype=np.float32)
    adds_kernel_magic[gpu_backend](
        KernelVec(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        KernelVec(4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), out)
    assert float(out[0]) == pytest.approx(5.0)
