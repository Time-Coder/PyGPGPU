"""The nesting tests, split out of `test_kernel_object_arguments.py`.

The subject is nesting.
"""
from __future__ import annotations

from _helpers.test_kernel_object_arguments_helpers import (
    Functions,
    Greeter,
    Holder,
    Pair,
    WithArray,
    WithImage,
    paints_an_image,
    reads_a_nested_object,
    reads_an_array_member,
    reads_nothing,
    writes_an_array_member_element,
    writes_to_the_object,
)

import numpy as np
import pytest


def test_a_nested_object_becomes_a_nested_structure(runner):
    """`h.inner.a` turns `inner` into a nested structure; the value comes through it."""
    out = np.zeros(1, dtype=np.float32)
    runner(reads_a_nested_object, Holder(Pair(6.0, 2.0)), out)
    assert float(out[0]) == pytest.approx(6.0)


def test_an_array_member_travels_as_its_own_parameter(gpu_backend, runner):
    """`w.xs` is an ndarray member: it leaves the synthesized structure and travels
    as its own kernel parameter, so the read answers the caller's array's bytes."""
    xs = np.arange(4, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    runner(reads_an_array_member, WithArray(xs), out)
    assert float(out[0]) == pytest.approx(float(xs[0]))


def test_an_array_member_element_write_lands_in_the_callers_array(gpu_backend, runner):
    """`item.xs[i] = v` stores through the array parameter -- the launch passed the
    caller's array itself, so the store lands in it with no write-back step."""
    xs = np.zeros(4, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    runner(writes_an_array_member_element, WithArray(xs), out)
    np.testing.assert_allclose(xs, (9.0, 0.0, 0.0, 0.0))


def test_an_array_member_sizes_the_dispatch_and_takes_tuple_stores(gpu_backend, runner):
    """`item.image.shape[0]` sizes the work-item domain from the member's own array
    parameter (the same holder the read path binds to), and the tuple store the
    examples spell (`item.image[i, j, 0] = v`) lands per component."""
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    out = np.zeros(1, dtype=np.float32)
    runner(paints_an_image, WithImage(image), out)
    np.testing.assert_array_equal(image[0], ((0, 128, 64), (0, 128, 64)))
    np.testing.assert_array_equal(image[1], ((255, 128, 64), (255, 128, 64)))


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


def test_a_function_member_is_refused():
    """The member is only reached when the body reads it, which this body does not -- so the
    launch succeeds and the refusal never fires. A body that did read it would be refused with
    the type's name."""
    from pygpgpu.kernel.objects import member_type

    with pytest.raises(SyntaxError) as caught:
        member_type(Functions().transform, "transform")
    assert "function" in str(caught.value)
