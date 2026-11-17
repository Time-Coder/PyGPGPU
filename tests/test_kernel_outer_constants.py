"""Module-level int/float/bool constants read by a kernel body, directly or via a helper.

Two layers carry an outer name, and the constants exercise both:

* **direct** -- a name the body reads itself is inlined by `_visit_name` (an int gets the
  integer-fits check, a float/bool the literal spelling of `_visit_constant`);
* **helper-mediated** -- the constant is read inside a helper the body calls, so it
  travels as an outer-scope *parameter*: the first translation requires it on the caller
  (`_require_outer_parameter`), the call site forwards it, and the signature declares it.

The second layer had a hole: the registration ran only on the *cache-miss* path of
`_emit_device_call`. A caller that hit a cached callee entry -- the second arm of a
diamond-shaped helper call, inside one kernel's translation -- forwarded the callee's
outer arguments without ever declaring the parameters, so the generated unit read `H_R`
(or whatever the constant is) that nothing defined, and the device compiler answered
`identifier "H_R" is undefined`. The hit path now requires the stored callee entries the
same way the miss path does; `diamond_caller` below is exactly that shape, and the other
tests pin the plain direct/helper spellings for all three scalar kinds.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# Direct references: the body reads the constant itself
# ---------------------------------------------------------------------------

DIRECT_GAIN: float = 2.5
DIRECT_COUNT: int = 4
DIRECT_FLAG: bool = True


@kernel
def direct_float(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i] * DIRECT_GAIN


@kernel
def direct_int(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i] + DIRECT_COUNT


@kernel
def direct_bool(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        if DIRECT_FLAG:
            out[i] = x[i] * 2.0
        else:
            out[i] = x[i] * 0.5


def test_direct_float_constant(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(direct_float, x, out)
    np.testing.assert_allclose(out, x * 2.5)


def test_direct_int_constant(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(direct_int, x, out)
    np.testing.assert_allclose(out, x + 4.0)


def test_direct_bool_constant(runner):
    x = np.arange(1, 5, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(direct_bool, x, out)
    np.testing.assert_allclose(out, x * 2.0)


# ---------------------------------------------------------------------------
# Helper-mediated references: the helper reads the constant
# ---------------------------------------------------------------------------

HELPER_FACTOR: float = 3.0
HELPER_BIAS: int = 2
HELPER_SWITCH: bool = False


@kernel
def helper_float(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = scale_by_factor(x[i])


@kernel
def helper_int(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = bias_by_count(x[i])


@kernel
def helper_bool(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = pick_by_flag(x[i])


def scale_by_factor(value: float) -> float:
    return value * HELPER_FACTOR


def bias_by_count(value: float) -> float:
    return value + HELPER_BIAS


def pick_by_flag(value: float) -> float:
    if HELPER_SWITCH:
        return value * 10.0
    return value * 0.25


def test_helper_float_constant(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(helper_float, x, out)
    np.testing.assert_allclose(out, x * 3.0)


def test_helper_int_constant(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(helper_int, x, out)
    np.testing.assert_allclose(out, x + 2.0)


def test_helper_bool_constant(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(helper_bool, x, out)
    np.testing.assert_allclose(out, x * 0.25)


# ---------------------------------------------------------------------------
# The diamond: a cached callee whose outer constant the second caller forwards
# ---------------------------------------------------------------------------

# The device-function cache is one translation unit's (`_compile`'s), so the hit happens
# *inside one kernel's* translation: two helpers of the same body call one leaf, and the
# second caller of the leaf meets its finished cache entry. This is exactly the shape the
# rendering project hit -- `sky_radiance` translated `density_at` first, and
# `optical_depth`, reaching the same leaf, hit the cached entry -- and the hit path never
# required the leaf's outer constant on the second caller, which then emitted a call to a
# parameter it had never declared.

DIAMOND_STEP: float = 5.0


def diamond_leaf(value: float) -> float:
    """The shared callee: the only place the constant is read."""
    return value * DIAMOND_STEP


def diamond_first_arm(value: float) -> float:
    """Translates the leaf first and builds its cache entry."""
    return diamond_leaf(value) + 1.0


def diamond_second_arm(value: float) -> float:
    """Hits the cached leaf entry, and forwards its constant to it."""
    return diamond_leaf(value) + 2.0


@kernel
def diamond_caller(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        a = diamond_first_arm(x[i])
        b = diamond_second_arm(x[i])
        out[i] = a + b


def test_a_cached_callee_declares_what_the_second_caller_forwards():
    """The second caller of a cached leaf must require the leaf's outer constant.

    Without the hit-path requirement, `diamond_second_arm` forwarded `DIAMOND_STEP` to the
    cached `diamond_leaf` without declaring the parameter: its rendered signature carries
    no `DIAMOND_STEP` while its body passes it, and the device compiler refuses the unit
    (the rendering project's `identifier "H_R" is undefined`). The assertion reads the
    translator's rendered device function directly, so it pins the signature itself --
    and the end-to-end launch below proves the whole unit still answers correctly.
    """
    from pygpgpu.kernel.translator import FunctionTranslator
    from pygpgpu.kernel.decorator import infer_ctype

    raw = diamond_caller.raw_func
    argument_types = [infer_ctype(np.zeros(4, dtype=np.float32)),
                      infer_ctype(np.zeros(4, dtype=np.float32))]
    translator = FunctionTranslator(raw, argument_types, "cuda",
                                    lambda name: raw.__globals__.get(name), {},
                                    is_kernel=True)
    translator.translate()
    second_arm = next(entry for entry in translator.device_cache.values()
                      if isinstance(entry, dict)
                      and "diamond_second_arm_" in str(entry.get("c_name", "")))
    # The signature declares the constant...
    assert "float DIAMOND_STEP" in second_arm["code"], second_arm["code"]
    # ...and the body forwards it to the leaf (the parameter plus the forwarded use).
    assert second_arm["code"].count("DIAMOND_STEP") >= 2, second_arm["code"]


def test_a_diamond_kernel_launches_and_answers(runner):
    """End to end: the whole diamond compiles and answers both arms' sum."""
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(diamond_caller, x, out)
    np.testing.assert_allclose(out, (x * 5.0 + 1.0) + (x * 5.0 + 2.0))


def test_a_constant_value_change_is_seen_on_the_next_call(runner, monkeypatch):
    """A read-only outer constant is a parameter, so the launch reads it live."""
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(helper_float, x, out)
    np.testing.assert_allclose(out, x * 3.0)

    monkeypatch.setattr("test_kernel_outer_constants.HELPER_FACTOR", 7.0)
    out = np.zeros(4, dtype=np.float32)
    runner(helper_float, x, out)
    np.testing.assert_allclose(out, x * 7.0)
