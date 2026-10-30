"""Arbitrary python objects as structure arguments.

An object argument is turned into a structure at launch, and the structure carries exactly the
members the kernel body reads -- a class with fifty attributes contributes the two its reader
mentions. The member set comes from a static scan of the kernel and the helpers it calls, and
the types come from the values the caller passed, so the interpreter and the GPU see the same
thing by construction.

The rules this file pins down:

* only the read members travel, and unused attributes are invisible;
* two objects of one class share a structure when their members can be typed consistently,
  with `int` widened to `float`; they stay two structures when one lacks a member the other
  needs or a member's types do not promote;
* `p.method()` is translated from the method found on the object's class;
* `helper(p)` puts the object into the helper's parameter, recursively;
* a nested object, an array member, a write, and an object nothing reads are each refused
  with the reason rather than guessed at.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


class Pair:
    """Two members, plus one nobody reads."""

    def __init__(self, a, b, unused="never read"):
        self.a = a
        self.b = b
        self.unused = unused


class Greeter:
    """A class with a method, which the body may call."""

    def __init__(self, base):
        self.base = base

    def doubled(self):
        return self.base * 2


class Fraction:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class Holder:
    """An object nested inside an object."""

    def __init__(self, inner):
        self.inner = inner


class WithArray:
    def __init__(self, xs):
        self.xs = xs


# ---------------------------------------------------------------------------
# The basics, on every backend
# ---------------------------------------------------------------------------


@kernel
def reads_two_members(item, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = item.a * item.b


@kernel
def reads_an_alias(item, out: np.ndarray):
    """`q = item; q.a` must count as a read of `item.a`."""
    for i in range(out.shape[0]):
        q = item
        out[i] = q.a * 2.0


@kernel
def calls_a_method(g, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = g.doubled()


def read_base(obj):
    return obj.base * 3.0


@kernel
def through_a_helper(g, out: np.ndarray):
    """A helper that receives the object reads its members too."""
    for i in range(out.shape[0]):
        out[i] = read_base(g)


def test_the_read_members_travel(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(reads_two_members, Pair(3.0, 5.0), out)
    assert float(out[0]) == pytest.approx(15.0)


def test_an_alias_counts_as_a_read(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(reads_an_alias, Pair(7.0, 0.0), out)
    assert float(out[0]) == pytest.approx(14.0)


def test_a_method_on_the_object(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(calls_a_method, Greeter(1.5), out)
    assert float(out[0]) == pytest.approx(3.0)


def test_a_helper_receiving_the_object(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(through_a_helper, Greeter(2.5), out)
    assert float(out[0]) == pytest.approx(7.5)


def test_unused_attributes_are_invisible():
    """Only `a` and `b` are read, so the synthesized structure has two fields -- the fact that
    `unused` exists, holding a string no device type could hold, must not matter."""
    from pygpgpu.kernel.objects import scan_object

    scan = scan_object(reads_two_members.raw_func, {"item": Pair(1.0, 2.0)},
                       lambda name: None)
    assert scan.members["item"] == ["a", "b"]


# ---------------------------------------------------------------------------
# Merging two objects of one class
# ---------------------------------------------------------------------------


@kernel
def two_objects(left, right, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = left.a + right.b


@kernel
def widens_across_instances(one, two, out: np.ndarray):
    """`one.a` is an int and `two.a` a float: the shared structure must answer float."""
    for i in range(out.shape[0]):
        out[i] = one.a + two.a


def test_two_objects_of_one_class(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(two_objects, Pair(1.0, 2.0), Pair(10.0, 20.0), out)
    assert float(out[0]) == pytest.approx(21.0)


def test_int_widens_to_float(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(widens_across_instances, Pair(1, 1.0), Pair(2.5, 2.0), out)
    assert float(out[0]) == pytest.approx(3.5)


def test_two_instances_stay_two_structures_when_one_lacks_a_member():
    """`left` only needs `a`, `right` needs `b` which `left` does not carry: merging would
    invent a member one of the two does not have, so they keep separate structures."""
    from pygpgpu.kernel.objects import convert_object_arguments

    class Lopsided:
        def __init__(self, **members):
            for name, value in members.items():
                setattr(self, name, value)

    @kernel
    def differing(left, right, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = left.a + right.b

    converted = convert_object_arguments(
        differing.raw_func, (Lopsided(a=1.0), Lopsided(b=2.0)), lambda name: None)
    assert type(converted[0]).__name__ != type(converted[1]).__name__


def test_two_instances_stay_two_structures_when_types_do_not_promote():
    """`a` is a float in one instance and a bool in the other: no type holds both honestly."""
    from pygpgpu.kernel.objects import convert_object_arguments

    converted = convert_object_arguments(
        widens_across_instances.raw_func, (Pair(1.5, 2.0), Pair(True, 2.0)), lambda name: None)
    assert type(converted[0]) is not type(converted[1])


# ---------------------------------------------------------------------------
# The refusals, each with its reason
# ---------------------------------------------------------------------------


@kernel
def reads_a_nested_object(h, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = h.inner.a


@kernel
def reads_an_array_member(w, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = w.xs[0]


@kernel
def writes_to_the_object(g, out: np.ndarray):
    """A single writer, so the member store is deterministic on every backend."""
    for i in range(out.shape[0]):
        if i == 0:
            g.base = 5.0
        out[i] = 1.0


@kernel
def reads_nothing(obj, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0


class Functions:
    def __init__(self):
        self.transform = lambda x: x


@kernel
def reads_a_function_member(f, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0


def test_a_nested_object_becomes_a_nested_structure(runner):
    """`h.inner.a` turns `inner` into a nested structure; the value comes through it."""
    out = np.zeros(1, dtype=np.float32)
    runner(reads_a_nested_object, Holder(Pair(6.0, 2.0)), out)
    assert float(out[0]) == pytest.approx(6.0)


def test_a_nested_read_through_a_non_object_is_refused(gpu_backend, runner):
    """w.xs[0] reads through an array member, which is not an object."""
    with pytest.raises(SyntaxError) as caught:
        runner(reads_an_array_member,
               WithArray(np.arange(4, dtype=np.float32)), np.zeros(1, np.float32))
    assert "array" in str(caught.value)


def test_an_array_member_is_refused(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(reads_an_array_member,
               WithArray(np.arange(4, dtype=np.float32)), np.zeros(1, np.float32))
    assert "array" in str(caught.value)


def test_a_write_travels_back_to_the_object(runner):
    """v3: a member store lands back on the caller's object after the launch."""
    greeter = Greeter(1.0)
    out = np.zeros(2, dtype=np.float32)
    runner(writes_to_the_object, greeter, out)
    assert float(greeter.base) == pytest.approx(5.0)


def test_an_object_nothing_reads_is_refused(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(reads_nothing, Greeter(1.0), np.zeros(1, np.float32))
    assert "nothing to put in a structure" in str(caught.value)


# ---------------------------------------------------------------------------
# v2: nested objects and lists of objects
# ---------------------------------------------------------------------------


class Wallet:
    """An object holding another object, plus a scalar of its own."""

    def __init__(self, inner, factor):
        self.inner = inner
        self.factor = factor


@kernel
def reads_through_two_levels(w, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = w.inner.a * w.factor


@kernel
def sums_a_list(items, out: np.ndarray):
    """A list of objects, read by index: the list becomes a record array."""
    for i in range(out.shape[0]):
        total: float = 0.0
        for j in range(len(items)):
            total += items[j].a
        out[i] = total


@kernel
def calls_methods_on_a_list(items, out: np.ndarray):
    for i in range(out.shape[0]):
        total: float = 0.0
        for j in range(len(items)):
            total += items[j].doubled()
        out[i] = total


def test_a_nested_object_becomes_a_nested_structure_deep(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(reads_through_two_levels, Wallet(Pair(3.0, 9.0), 2.0), out)
    assert float(out[0]) == pytest.approx(6.0)


def test_a_list_of_objects_becomes_a_record_array(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(sums_a_list, [Pair(1.0, 0.0), Pair(2.0, 0.0), Pair(3.5, 0.0)], out)
    assert float(out[0]) == pytest.approx(6.5)


def test_methods_called_on_list_elements(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(calls_methods_on_a_list, [Greeter(1.0), Greeter(2.5)], out)
    assert float(out[0]) == pytest.approx(7.0)


def test_a_mixed_class_list_is_refused():
    from pygpgpu.kernel.objects import convert_object_arguments

    class Other:
        def __init__(self):
            self.a = 1.0

    with pytest.raises(SyntaxError) as caught:
        convert_object_arguments(sums_a_list.raw_func,
                                 ([Pair(1.0, 2.0), Other()],), lambda name: None)
    assert "2 classes" in str(caught.value)


def test_an_empty_list_is_refused():
    from pygpgpu.kernel.objects import convert_object_arguments

    with pytest.raises(SyntaxError) as caught:
        convert_object_arguments(sums_a_list.raw_func, ([],), lambda name: None)
    assert "empty list" in str(caught.value)


def test_a_write_to_a_list_element_is_refused():
    from pygpgpu.kernel.objects import convert_object_arguments

    @kernel
    def writes_to_an_element(items, out: np.ndarray):
        for i in range(out.shape[0]):
            items[0].a = 2.0
            out[i] = 1.0

    with pytest.raises(SyntaxError) as caught:
        convert_object_arguments(writes_to_an_element.raw_func,
                                 ([Pair(1.0, 2.0)],), lambda name: None)
    assert "list of objects" in str(caught.value)


# ---------------------------------------------------------------------------
# v3: member writes travel back to the caller's object
# ---------------------------------------------------------------------------


def test_a_write_travels_back_through_a_propertyless_object(runner):
    greeter = Greeter(1.0)
    out = np.zeros(2, dtype=np.float32)
    runner(writes_to_the_object, greeter, out)
    assert float(greeter.base) == pytest.approx(5.0)


def test_a_write_to_a_property_is_refused():
    from pygpgpu.kernel.objects import convert_object_arguments

    class WithProperty:
        def __init__(self):
            self._x = 1.0

        @property
        def x(self):
            return self._x

    @kernel
    def writes_a_property(p, out: np.ndarray):
        for i in range(out.shape[0]):
            p.x = 2.0
            out[i] = 1.0

    with pytest.raises(SyntaxError) as caught:
        convert_object_arguments(writes_a_property.raw_func, (WithProperty(),),
                                 lambda name: None)
    assert "property" in str(caught.value)


def test_a_write_outside_slots_is_refused():
    from pygpgpu.kernel.objects import convert_object_arguments

    class Slotted:
        __slots__ = ("a",)

        def __init__(self):
            self.a = 1.0

    @kernel
    def writes_outside_slots(p, out: np.ndarray):
        for i in range(out.shape[0]):
            p.missing = 2.0
            out[i] = 1.0

    with pytest.raises(SyntaxError) as caught:
        convert_object_arguments(writes_outside_slots.raw_func, (Slotted(),),
                                 lambda name: None)
    assert "__slots__" in str(caught.value)

def test_a_function_member_is_refused():
    """The member is only reached when the body reads it, which this body does not -- so the
    launch succeeds and the refusal never fires. A body that did read it would be refused with
    the type's name."""
    from pygpgpu.kernel.objects import member_type

    with pytest.raises(SyntaxError) as caught:
        member_type(Functions().transform, "transform")
    assert "function" in str(caught.value)


def test_a_member_missing_from_the_instance_names_it():
    from pygpgpu.kernel.objects import _fields_for

    class OnlyA:
        def __init__(self):
            self.a = 1.0

    with pytest.raises(SyntaxError) as caught:
        _fields_for(OnlyA(), ["a", "b"])
    assert "has no attribute 'b'" in str(caught.value)
