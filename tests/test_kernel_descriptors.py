"""A class attribute that computes itself: a descriptor's `__get__`/`__set__` on the device.

Python resolves a class-body descriptor before it looks for a field: reading `box.total` runs
`type(box).total.__get__(box, type(box))`, storing runs `__set__`. `property` is the one the
translator already answered with a direct getter/setter call; these tests pin the general case
for every shape a kernel can hold a class value in -- a structure argument, the receiver's own
class, and a class-typed member of the receiver.

The descriptor itself and its `owner` class are *host* objects, so they are skipped when the
accessor is translated. A descriptor that reads either is refused by name: it has no device
value to read, and the tests check that the message says which parameter it was.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from pygpgpu import kernel


class Doubled:
    """Computes from the instance and stores through it -- the translatable shape."""

    def __get__(self, instance, owner=None):
        return instance.a * 10.0

    def __set__(self, instance, value):
        instance.a = value + 1.0


class Stateful:
    """Keeps its factor in itself: the descriptor object is a host object."""

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __get__(self, instance, owner=None):
        return instance.a * self.factor


class OwnerReader:
    """Reads its owner class, which a device cannot hold."""

    def __get__(self, instance, owner=None):
        return instance.a + float(len(owner.__name__))


class SetterOnly:
    """A data descriptor without `__get__`: python would not read through it either."""

    def __set__(self, instance, value):
        instance.a = value


class Box(ctypes.Structure):
    """Eleven floats: a structure layout (no vector type has this width)."""

    _fields_ = [(name, ctypes.c_float)
                for name in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k")]
    doubled = Doubled()
    stateful = Stateful(3.0)
    owner_read = OwnerReader()
    set_only = SetterOnly()


@kernel
def round_trips(box, out: np.ndarray):
    for i in range(out.shape[0]):
        box.doubled = 5.0
        out[i] = box.doubled


@kernel
def accumulates_through_the_descriptor(box, out: np.ndarray):
    for i in range(out.shape[0]):
        box.doubled += 2.0
        out[i] = box.doubled


@kernel
def reads_stateful(box, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = box.stateful


@kernel
def reads_owner(box, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = box.owner_read


@kernel
def reads_setter_only(box, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = box.set_only


def test_a_descriptor_is_a_device_call(runner):
    """`box.doubled = 5` runs `__set__` and `box.doubled` runs `__get__`."""
    box = Box(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0)
    out = np.zeros(1, dtype=np.float32)
    runner(round_trips, box, out)
    assert float(out[0]) == pytest.approx(60.0)     # a = 6, then 6 * 10
    assert float(box.a) == pytest.approx(6.0)


def test_an_in_place_operator_reads_and_writes_through_it(runner):
    """`box.doubled += 2` is `box.doubled = box.doubled + 2` -- python's own rule."""
    box = Box(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    out = np.zeros(1, dtype=np.float32)
    runner(accumulates_through_the_descriptor, box, out)
    assert float(out[0]) == pytest.approx(130.0)    # (10 + 2) -> a = 13 -> 130
    assert float(box.a) == pytest.approx(13.0)


def test_a_descriptor_that_reads_its_own_state_is_refused(gpu_backend):
    """The descriptor object is a host attribute; there is nothing on the device to read."""
    with pytest.raises(SyntaxError) as caught:
        reads_stateful[gpu_backend](
            Box(*([1.0] * 11)), np.zeros(1, dtype=np.float32))
    message = str(caught.value)
    assert "stateful" in message and "descriptor object itself" in message


def test_a_descriptor_that_reads_its_owner_is_refused(gpu_backend):
    with pytest.raises(SyntaxError) as caught:
        reads_owner[gpu_backend](Box(*([1.0] * 11)), np.zeros(1, dtype=np.float32))
    message = str(caught.value)
    assert "owner_read" in message and "owner class" in message


def test_a_setter_only_descriptor_is_not_a_read(gpu_backend):
    """No `__get__` means python does not answer a read through it, and neither does the
    device: the name then falls to the ordinary member lookup, which refuses it."""
    with pytest.raises(SyntaxError) as caught:
        reads_setter_only[gpu_backend](Box(*([1.0] * 11)), np.zeros(1, dtype=np.float32))
    assert "set_only" in str(caught.value)


# ---------------------------------------------------------------------------
# The receiver, and a class-typed member of it
# ---------------------------------------------------------------------------


class Holder:
    """A kernel whose own class carries a descriptor, and a class-typed member."""

    doubled = Doubled()
    a = 2.0

    def __init__(self) -> None:
        self.a = 2.0
        self.cell = Box(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @kernel
    def reads_own(self, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = self.doubled

    @kernel
    def writes_own(self, value, out: np.ndarray):
        for i in range(out.shape[0]):
            self.doubled = value
            out[i] = self.doubled

    @kernel
    def bumps_own(self, out: np.ndarray):
        for i in range(out.shape[0]):
            self.doubled += 3.0
            out[i] = self.doubled

    @kernel
    def reads_member(self, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = self.cell.doubled

    @kernel
    def writes_member(self, value, out: np.ndarray):
        for i in range(out.shape[0]):
            self.cell.doubled = value
            out[i] = self.cell.a


def test_a_receiver_descriptor_is_the_kernel_class_own(runner):
    """`self.doubled` runs the descriptor, and the member it reads is uploaded with it."""
    holder = Holder()
    out = np.zeros(1, dtype=np.float32)
    runner(holder.reads_own, out)
    assert float(out[0]) == pytest.approx(20.0)     # a = 2, then 2 * 10

    holder = Holder()
    runner(holder.writes_own, 4.0, out)
    assert float(out[0]) == pytest.approx(50.0)     # a = 5, then 5 * 10
    assert float(holder.a) == pytest.approx(5.0)

    holder = Holder()
    runner(holder.bumps_own, out)
    assert float(out[0]) == pytest.approx(240.0)    # (20 + 3) -> a = 24 -> 240
    assert float(holder.a) == pytest.approx(24.0)


def test_a_class_typed_member_is_indexed_by_its_descriptor(runner):
    """`self.cell.doubled` reads through the member's class, write and read alike."""
    holder = Holder()
    out = np.zeros(1, dtype=np.float32)
    runner(holder.reads_member, out)
    assert float(out[0]) == pytest.approx(10.0)     # cell.a = 1, then 1 * 10

    holder = Holder()
    runner(holder.writes_member, 6.0, out)
    assert float(out[0]) == pytest.approx(7.0)      # cell.a = 7
    assert float(holder.cell.a) == pytest.approx(7.0)
