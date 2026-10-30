"""Method calls on a structure value, and the scope a helper resolves its names in.

`ray.hit_sphere(center, radius)` is a call on a *value* whose class defines the method. There
is nothing new to translate: a method is an ordinary python function whose first parameter is
the receiver, which is what `self` already is in a device helper. So the call becomes a device
function with the receiver passed first, and the body reads `self.origin` exactly as any
structure-typed parameter would.

That only works when the class is reachable, which is why a bare record dtype is refused: two
classes with one layout share a dtype, so the dtype cannot say whose methods to compile.

The second half of this file is about the scope such a method is translated in. Until a
structure's methods lived in *another* module than the kernel, a helper could be translated
against the kernel's globals and nothing noticed; `ray.py` importing `dot` for itself is the
first case where that is wrong.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest
from pygf import float3, point3f

import helper_scope_module
from pygpgpu import kernel


class Ray(ctypes.Structure):
    _fields_ = [("origin", point3f), ("direction", float3)]

    def at(self, t: float) -> point3f:
        """A method that reads its own members."""
        return self.origin + t * self.direction

    def hit_sphere(self, center: point3f, radius: float) -> bool:
        """The shape the example uses: a vector parameter and a scalar end to end."""
        oc: float3 = center - self.origin
        a: float = dot(self.direction, self.direction)
        b: float = -2.0 * dot(self.direction, oc)
        c: float = dot(oc, oc) - radius**2
        discriminant: float = b**2 - 4 * a * c
        return discriminant >= 0


# The module-level import a method body may rely on, like `ray.py` does.
from pygf import dot  # noqa: E402


@kernel
def calls_a_method(ray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = ray.at(2.0)["x"]


@kernel
def calls_a_method_taking_arguments(ray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if ray.hit_sphere(point3f(0.0, 0.0, -2.0), 1.0) else 0.0


@kernel
def calls_a_method_on_a_constructed_value(out: np.ndarray):
    """The receiver does not have to be a parameter; a constructed one works too."""
    for i in range(out.shape[0]):
        ray = Ray(point3f(0.0, 0.0, 0.0), float3(0.0, 0.0, -1.0))
        out[i] = 1.0 if ray.hit_sphere(point3f(0.0, 0.0, -2.0), 1.0) else 0.0


@kernel
def method_through_a_helper(ray, out: np.ndarray):
    """A method called from a device helper is one more forwarding step."""
    for i in range(out.shape[0]):
        out[i] = distance_to_ray(ray)


def distance_to_ray(ray):
    return ray.at(1.0)["y"]


def make_ray(origin=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)) -> Ray:
    return Ray(point3f(*origin), float3(*direction))


def test_a_method_reading_its_own_members(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(calls_a_method, make_ray(), out)
    assert float(out[0]) == pytest.approx(2.0)


def test_a_method_taking_arguments(runner):
    """A ray down -z from the origin hits a unit sphere centred at (0, 0, -2)."""
    out = np.zeros(1, dtype=np.float32)
    runner(calls_a_method_taking_arguments, make_ray(direction=(0.0, 0.0, -1.0)), out)
    assert float(out[0]) == pytest.approx(1.0)

    out = np.zeros(1, dtype=np.float32)
    runner(calls_a_method_taking_arguments, make_ray(direction=(1.0, 0.0, 0.0)), out)
    assert float(out[0]) == pytest.approx(0.0)


def test_a_method_on_a_value_constructed_in_the_body(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(calls_a_method_on_a_constructed_value, out)
    assert float(out[0]) == pytest.approx(1.0)


def test_a_method_called_from_a_helper(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(method_through_a_helper, make_ray(origin=(0.0, 5.0, 0.0)), out)
    assert float(out[0]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# The class has to be reachable, and a record dtype has no methods to find
# ---------------------------------------------------------------------------


RECORD = np.dtype([("i", "<i4"), ("f", "<f4")])


@kernel
def calls_a_method_on_a_bare_record(rows: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = rows[i].size_of_thing()


def test_a_record_dtype_has_no_class_to_take_methods_from(gpu_backend, runner):
    """A bare `np.dtype` declares a layout and nothing else, so there is no class to
    translate a method from. (The layout may also have been declared by some `ctypes`
    class earlier in the session, in which case that class is consulted and the answer is
    the other clear error: the member it names is not a method.)"""
    with pytest.raises(SyntaxError) as caught:
        runner(calls_a_method_on_a_bare_record, np.zeros(2, dtype=RECORD),
               np.zeros(1, dtype=np.float32))
    assert "size_of_thing" in str(caught.value)


@kernel
def calls_an_unknown_method(ray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = ray.reflect_at(1.0)


def test_an_unknown_method_says_what_the_class_has(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(calls_an_unknown_method, make_ray(), np.zeros(1, dtype=np.float32))
    assert "reflect_at" in str(caught.value)


# ---------------------------------------------------------------------------
# The scope a helper is translated in
# ---------------------------------------------------------------------------


@kernel
def calls_a_helper_in_another_module(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = helper_scope_module.scale_by_its_own_global(2.0)


MODULE_GLOBAL = 7.0


def scaled_by_the_kernel_module(value):
    """Reads a global of *this* module, which the call site has to forward."""
    return value * MODULE_GLOBAL


@kernel
def calls_a_helper_reading_the_kernel_modules_global(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = scaled_by_the_kernel_module(3.0)


def test_a_helper_resolves_its_own_modules_names(runner):
    """The helper's module imports what its body calls; the kernel's module need not."""
    out = np.zeros(1, dtype=np.float32)
    runner(calls_a_helper_in_another_module, out)
    assert float(out[0]) == pytest.approx(helper_scope_module.SCALE * 2.0)


def test_a_helper_still_sees_the_kernel_modules_outer_names(runner):
    """Falling back to the caller's resolver is what keeps outer variables working."""
    out = np.zeros(1, dtype=np.float32)
    runner(calls_a_helper_reading_the_kernel_modules_global, out)
    assert float(out[0]) == pytest.approx(3.0 * MODULE_GLOBAL)
