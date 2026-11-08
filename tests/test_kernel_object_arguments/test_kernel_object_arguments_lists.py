"""The lists tests, split out of `test_kernel_object_arguments.py`.

The subject is lists.
"""
from __future__ import annotations

from _helpers.test_kernel_object_arguments_helpers import (
    Greeter,
    Pair,
    Wallet,
    calls_methods_on_a_list,
    reads_through_two_levels,
    sums_a_list,
)

import numpy as np
import pytest
from pygpgpu import kernel


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
