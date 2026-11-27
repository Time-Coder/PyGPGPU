"""`with inline, kernel(...)` inside a method: what `self` and `cls` mean.

A block written in a method reads the receiver out of the enclosing frame like any other
local, and it is the one capture that travels **separately**: `captured_names` hoists it to
the front, and the launch hands it to the kernel as a receiver rather than as an argument.
`PyKernel` decides that from the parameter's name -- `self` for an instance method, `cls`
for a classmethod -- which is python's own spelling for the two.

Two things are pinned here, because they are what a reader of the block will assume:

* a member the block **writes** reaches the caller: the receiver travels as one structure
  whose read-back is copied home, and an ndarray member is its own argument (so it comes
  back through the ordinary buffer read-back);
* `cls` is the class the method was *called on*, not the one it was defined on -- python's
  binding rule -- so a subclass call reads the subclass's attributes and writes them back
  onto the subclass, leaving the base class alone.

A receiver spelled something else (`this`) is not special: it is captured as an ordinary
object argument, which still reads and writes, but it is not followed as a method receiver
(`this.helper()` is not resolved as one).
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import inline, kernel


def _close(actual:object, expected:object)->None:
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


class _Box:
    """A method whose block captures the receiver, which travels as a structure."""

    def __init__(self, factor:float)->None:
        self.factor:float = factor

    def scale_into(self, a:np.ndarray, out:np.ndarray, backend:str)->None:
        with inline, kernel(default_device=backend):
            for i in range(out.shape[0]):
                out[i] = a[i] * self.factor


def test_a_captured_object_reaches_the_device_as_a_structure(backend)->None:
    box:_Box = _Box(3.0)
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    box.scale_into(a, out, backend)
    _close(out, a * 3.0)


def test_the_receiver_is_not_a_general_capture(backend)->None:
    """A `self` that is not the enclosing method's receiver is just another name."""
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    box:_Box = _Box(2.0)
    with inline, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = a[i] * box.factor
    _close(out, a * 2.0)


class _Writer:
    """A method whose block writes the receiver: a scalar member and an array member."""

    def __init__(self, factor:float, size:int)->None:
        self.factor:float = factor
        self.first:float = -1.0
        self.count:int = 0
        self.buf:np.ndarray = np.zeros(size, dtype=np.float32)

    def write_into(self, a:np.ndarray, out:np.ndarray, backend:str)->None:
        with inline, kernel(default_device=backend):
            for i in range(out.shape[0]):
                out[i] = a[i] * self.factor
                self.buf[i] = a[i] + self.factor
                if i == 0:
                    self.first = a[0]
                    self.count = out.shape[0]


def test_a_member_written_in_a_block_reaches_the_instance(backend)->None:
    """`self.<attr> = ...` writes through the receiver struct, and the read-back lands."""
    a:np.ndarray = np.arange(1, 5, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    writer:_Writer = _Writer(2.0, 4)
    writer.write_into(a, out, backend)
    _close(out, a * 2.0)
    _close(writer.buf, a + 2.0)                 # an ndarray member is its own argument
    assert writer.first == pytest.approx(float(a[0]))
    assert writer.count == 4


class _ClassWriter:
    """A classmethod whose block reads `cls.factor` and writes class attributes."""

    factor:float = 3.0
    first:float = -1.0
    count:int = 0

    @classmethod
    def write_into(cls, a:np.ndarray, out:np.ndarray, backend:str)->None:
        with inline, kernel(default_device=backend):
            for i in range(out.shape[0]):
                out[i] = a[i] * cls.factor
                if i == 0:
                    cls.first = a[0]
                    cls.count = out.shape[0]


def test_a_classmethod_receiver_reads_and_writes_the_class(backend)->None:
    a:np.ndarray = np.arange(1, 5, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    _ClassWriter.factor = 3.0
    _ClassWriter.first = -1.0
    _ClassWriter.count = 0
    _ClassWriter.write_into(a, out, backend)
    _close(out, a * 3.0)
    assert _ClassWriter.first == pytest.approx(float(a[0]))
    assert _ClassWriter.count == 4


class _ClassBase:
    factor:float = 2.0
    first:float = -1.0

    @classmethod
    def write_into(cls, a:np.ndarray, out:np.ndarray, backend:str)->None:
        with inline, kernel(default_device=backend):
            for i in range(out.shape[0]):
                out[i] = a[i] * cls.factor
                if i == 0:
                    cls.first = a[0]


class _ClassSub(_ClassBase):
    factor:float = 10.0


def test_a_classmethod_receiver_is_the_class_it_was_called_on(backend)->None:
    """`cls` binds as python binds it: a subclass call reads and writes the subclass."""
    a:np.ndarray = np.arange(1, 5, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    _ClassBase.factor, _ClassBase.first = 2.0, -1.0
    _ClassSub.factor, _ClassSub.first = 10.0, -1.0
    _ClassSub.write_into(a, out, backend)
    _close(out, a * 10.0)
    assert _ClassSub.first == pytest.approx(float(a[0]))
    assert _ClassBase.first == -1.0             # the write is the subclass's attribute
