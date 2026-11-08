"""The classes and the fixtures the object-argument tests share.

A fixture has to be found by pytest, and a class is the SAME class whichever
group uses it -- two copies would be two classes, and the tests compare them.
"""
from __future__ import annotations

from __future__ import annotations
import numpy as np
from pygpgpu import kernel


class Fraction:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class Functions:
    def __init__(self):
        self.transform = lambda x: x


class Greeter:
    """A class with a method, which the body may call."""

    def __init__(self, base):
        self.base = base

    def doubled(self):
        return self.base * 2


class Holder:
    """An object nested inside an object."""

    def __init__(self, inner):
        self.inner = inner


class Pair:
    """Two members, plus one nobody reads."""

    def __init__(self, a, b, unused="never read"):
        self.a = a
        self.b = b
        self.unused = unused


class Range:
    """A plain class whose value a body hands straight to a helper."""

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def size(self):
        return self.high - self.low

    def contains(self, value):
        return self.low <= value


class Wallet:
    """An object holding another object, plus a scalar of its own."""

    def __init__(self, inner, factor):
        self.inner = inner
        self.factor = factor


class WithArray:
    def __init__(self, xs):
        self.xs = xs


@kernel
def calls_a_method(g, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = g.doubled()


@kernel
def calls_methods_on_a_list(items, out: np.ndarray):
    for i in range(out.shape[0]):
        total: float = 0.0
        for j in range(len(items)):
            total += items[j].doubled()
        out[i] = total


@kernel
def constructs_inline(out: np.ndarray):
    """`sum_of(Pair(1.0, 2.0))`: the value has no name of its own."""
    for i in range(out.shape[0]):
        out[i] = sum_of(Pair(1.0, 2.0))


@kernel
def constructs_named(out: np.ndarray):
    """The same value bound first, which already worked."""
    for i in range(out.shape[0]):
        pair = Pair(1.0, 2.0)
        out[i] = sum_of(pair)


def read_base(obj):
    return obj.base * 3.0


@kernel
def reads_a_function_member(f, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0


@kernel
def reads_a_nested_object(h, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = h.inner.a


@kernel
def reads_a_plain_objects_members(item, out: np.ndarray):
    """A kernel of its own, so this test's launch always writes the module's source."""
    for i in range(out.shape[0]):
        out[i] = item.a * item.b


@kernel
def reads_an_alias(item, out: np.ndarray):
    """`q = item; q.a` must count as a read of `item.a`."""
    for i in range(out.shape[0]):
        q = item
        out[i] = q.a * 2.0


@kernel
def reads_an_array_member(w, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = w.xs[0]


@kernel
def reads_nothing(obj, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0


@kernel
def reads_through_two_levels(w, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = w.inner.a * w.factor


@kernel
def reads_two_members(item, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = item.a * item.b


def sum_of(pair):
    """The helper the constructed value is handed to, like the example's `self.hit`."""
    return pair.a + pair.b


@kernel
def sums_a_list(items, out: np.ndarray):
    """A list of objects, read by index: the list becomes a record array."""
    for i in range(out.shape[0]):
        total: float = 0.0
        for j in range(len(items)):
            total += items[j].a
        out[i] = total


@kernel
def through_a_helper(g, out: np.ndarray):
    """A helper that receives the object reads its members too."""
    for i in range(out.shape[0]):
        out[i] = read_base(g)


@kernel
def two_objects(left, right, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = left.a + right.b


@kernel
def widens_across_instances(one, two, out: np.ndarray):
    """`one.a` is an int and `two.a` a float: the shared structure must answer float."""
    for i in range(out.shape[0]):
        out[i] = one.a + two.a


@kernel
def writes_to_the_object(g, out: np.ndarray):
    """A single writer, so the member store is deterministic on every backend."""
    for i in range(out.shape[0]):
        if i == 0:
            g.base = 5.0
        out[i] = 1.0
