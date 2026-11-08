"""The writes tests, split out of `test_kernel_object_arguments.py`.

The subject is writes.
"""
from __future__ import annotations

from _helpers.test_kernel_object_arguments_helpers import (
    Greeter,
    writes_to_the_object,
)

import numpy as np
import pytest
from pygpgpu import kernel


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


def test_a_member_missing_from_the_instance_names_it():
    from pygpgpu.kernel.objects import _fields_for

    class OnlyA:
        def __init__(self):
            self.a = 1.0

    with pytest.raises(SyntaxError) as caught:
        _fields_for(OnlyA(), ["a", "b"])
    assert "has no attribute 'b'" in str(caught.value)
