"""Regression tests about types: vector builtins, the hoist, widths, and ``%``.

Every test here once failed, on purpose: they were written as failing tests first (TDD),
the specification for their fixes, and they have passed since the fixes landed -- they stay
to keep each bug fixed. Each one names the repro that found it and what the right answer
is, and none of the answers invents a new rule -- each is what python (or numpy, where the
interpreter runs numpy) already computes for the same source. The kernels live one module
per bug in `regressions/` (one python module is one translation unit, so a compile error
would otherwise take every kernel in the module down with it). The object-shaped groups
live in `test_kernel_regressions_objects.py`.

These groups were found by porting the Shadertoy of the Week to
`examples/shadertoy/light_cycle.py` -- a ray-marched tunnel whose GLSL is ordinary shader
vocabulary -- and by the shadertoy viewer itself:

* **the vector-wide builtins**: ``abs``/``min``/``max`` on whole vectors were refused
  ("requires a numeric argument", "vector arithmetic is handled separately") although the
  scalar spellings work and the component arithmetic they expand to works;
* **``smoothstep``**: was not exported, and pygf's own could not be walked ("unknown name
  '_smoothstep'") -- a GLSL name with no device spelling at all;
* **the hoisted vector local**: a local first assigned inside a block was hoisted with the
  scalar zero-init ``(T)0``, which CUDA C++ refuses for a vector struct ("no suitable
  constructor exists to convert from int to float3") -- the record path already knew to
  use ``{0}``, the vector path fell between the two;
* **``math.pow`` width mixing**: a float32 vector component against a python exponent emitted
  ``pow(float, double)`` and no C++ overload matched ("more than one instance of overloaded
  function pow matches") -- python promotes to double and answers;
* **``%`` on floats**: the device emitted C's ``fmod`` (dividend's sign) where python floors
  by the divisor's sign -- ``-1.0 % 4.0`` is 3.0 in python and -1.0 on the device, so the
  same source answers differently per backend;
* **an override with extra parameters**: a hook whose signature does not match the base
  call loses the whole self-attribute scan -- an arity error for the mismatched shape, and
  a silent empty scan (``discovered attributes: image``) when the base call's expression
  reads ``self`` -- so every uniform the subclass reads disappears from the launch;
* **the receiver struct's layout**: a synthesized receiver whose members mix widths
  (float2 + float + int + float4, the promo viewer's uniform set) packed by `ctypes`
  alignment where the compiler aligns by the device rules, so every member after the
  first vector read the wrong bytes (group 15).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from regressions.attribute_layout_mix import ClassAndInstance, Float2BeforeInt, ViewerLike
from regressions.extra_arg_override import ExtraArg, FlipSin
from regressions.float_mod_floored import mod_four
from regressions.pow_width_mixing import MixedWidthPow
from regressions.smoothstep_builtin import fades_one_edge
from regressions.swizzle_store_aliasing import swap_through_swizzle
from regressions.vector_hoist_zero import HoistedVector
from regressions.vector_minmax_abs import vector_abs_and_minmax

# ---------------------------------------------------------------------------
# Group 9 -- the vector-wide builtins
# ---------------------------------------------------------------------------


def test_abs_min_and_max_expand_component_wise_over_vectors(gpu_backend):
    """``abs(v)``, ``max(v, s)`` and ``min(v, w)`` are the component-wise GLSL semantics.

    Each used to be refused at translation time ("abs() requires a numeric argument", "vector
    arithmetic is handled separately"). The scalar spellings work, and the component
    arithmetic these expand to works, so the emission is known: expand per component and
    emit, exactly like the +/* paths already do. The expected numbers are python's:
    ``abs((-1, 2, -3))`` sums to 6, ``max((1, -5, 3), 0)`` folded with ``min(, (2, 2, 2))``
    sums to 3, and ``abs(min((-2, 0.5), (1, 1)).x)`` is 2.
    """
    out = np.zeros(3, dtype=np.float64)
    vector_abs_and_minmax[gpu_backend](out)
    np.testing.assert_allclose(out, [6.0, 3.0, 2.0])


def test_smoothstep_interpolates_the_midpoint(gpu_backend):
    """``smoothstep(0.2, 0.6, 0.4)`` is 0.5 -- the clamp-and-fade every shader uses.

    Used to be: not exported from the package, and pygf's own was refused in translation with
    ``unknown name '_smoothstep'`` -- pygf's source calls a private helper the walk cannot
    resolve. The emission is three lines (the GLSL formula), scalar here and per component
    once the vector builtins land. The expected number is python's: t = (0.4 - 0.2) / 0.4 =
    0.5, and t * t * (3 - 2 * t) = 0.5.
    """
    out = np.zeros(1, dtype=np.float64)
    fades_one_edge[gpu_backend](np.float64(0.4), out)
    assert out[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Group 10 -- the hoisted vector local
# ---------------------------------------------------------------------------


def test_a_vector_local_assigned_in_a_block_answers_the_branch(gpu_backend):
    """A float3 local whose first assignment sits inside a branch must compile and answer.

    Used to be: the hoisted declaration was spelled ``(float3)0`` -- the scalar zero -- which CUDA
    C++ refuses ("no suitable constructor exists to convert from int to float3"). The record
    path already initialises hoisted declarations ``{0}`` for exactly this reason; the vector
    needs the same treatment (a ``make_float3(0, 0, 0)`` spelling, or declaring at first
    assignment). Both branches assign here, so the answer is the branch's colour, not the
    zero: red where ``uv.x > 0.25`` (columns 1 and 2), green in column 0 -- python's numbers
    for the same source.
    """
    viewer = HoistedVector()
    viewer.draw[gpu_backend]()
    expected = np.zeros((2, 3, 3), dtype=np.uint8)
    for i in range(2):
        for j in range(3):
            expected[i, j] = (255, 0, 0) if (j + 0.5) / 3.0 > 0.25 else (0, 255, 0)
    np.testing.assert_allclose(viewer.image, expected)


# ---------------------------------------------------------------------------
# Group 11 -- math.pow width mixing
# ---------------------------------------------------------------------------


def test_pow_widens_a_vector_component_to_double(gpu_backend):
    """``math.pow(v.x, 2.2)`` -- a float32 component against a python exponent -- is python's.

    Used to be: emitted as ``pow(float, double)`` and no C++ overload matched ("more than one
    instance of overloaded function pow matches"). python's math.pow is double throughout, so
    the device widens the float32 argument and emits ``pow(double, double)`` -- the
    two-double spelling already compiles, so this is purely the width-mixing case. The
    expected number is python's: 255 * 0.5 ** 2.2.
    """
    viewer = MixedWidthPow()
    viewer.draw[gpu_backend]()
    expected = 255.0 * math.pow(0.5, 2.2)
    assert viewer.image[0, 0] == pytest.approx(expected, abs=1.0)
    assert viewer.image[0, 1] == pytest.approx(expected, abs=1.0)


# ---------------------------------------------------------------------------
# Group 12 -- % on floats is fmod, not python's floored remainder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x, expected", [(-1.0, 3.0), (-5.5, 2.5)])
def test_float_mod_is_floored_like_python(gpu_backend, x, expected):
    """``x % 4.0`` floors by the divisor's sign on the host, takes the dividend's on the device.

    The emission used to be C's ``fmod``: ``-1.0 % 4.0`` answers -1.0 on the device where python
    answers 3.0, and ``-5.5 % 4.0`` answers -1.5 where python answers 2.5 -- the same source
    disagreeing between the interpreter backend and every device backend. The fix is the
    floored-mod emission python semantics owe (``x - y * floor(x / y)``) in the operands' own
    width. Positive operands already agree, which is why the Light Cycle port survives with
    positive-only operands -- but the rule is python's, not "works when the operand is nice".
    """
    out = np.zeros(1, dtype=np.float64)
    mod_four[gpu_backend](np.float64(x), out)
    assert out[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Group 13 -- an override with extra parameters loses the whole scan
# ---------------------------------------------------------------------------


def test_an_override_with_extra_parameters_collects_its_members(gpu_backend):
    """The extra-parameter shape must not silently drop the subclass's uniform reads.

    The launch used to surface an arity error (``sample() takes 5 device arguments but 4
    types were given``) -- honest, but it should not reach the launch at all: the base call
    does not match the override's signature, and that is a translation-time fact. The right
    answer is either a clear refusal naming the mismatch, or (the useful one) collecting the
    override's reads against the owner's signature and launching -- which is what the
    shadertoy viewer needs, since the base ``draw`` it inherits cannot know every port's
    ``mainImage`` signature. This test pins the useful answer: the launch succeeds and the
    override's members are read.
    """
    viewer = ExtraArg()
    viewer.draw[gpu_backend]()
    # sample(i, j, y) ignores y and answers sin(0.5) * (i + j).
    import math as _math

    expected = _math.sin(0.5) * np.array([[0, 1, 2], [1, 2, 3]], dtype=np.float32)
    np.testing.assert_allclose(viewer.image, expected, rtol=1e-5)


def test_a_self_reading_call_expression_keeps_the_override_scan(gpu_backend):
    """The flip shape: the base call's third argument is built from ``self.image.shape[0]``.

    The launch used to build without a single member the override reads -- ``discovered
    attributes: image`` -- and the refusal names the first missing one. The scan followed the
    call into the override and came back empty; the arity between the base call and the
    override matches here (both three), so the defect is the base call expression reading
    ``self`` while the scanned callee at decoration time was the base's own two-parameter
    hook. Whatever the fix resolves against, the override's members must reach the launch:
    ``origin`` (a float2) and ``phase`` (a double).
    """
    viewer = FlipSin()
    viewer.draw[gpu_backend]()
    # sample(i, j, y) answers sin(0.5) * (i / 3): row 0 is 0, row 1 is sin(0.5)/3.
    import math as _math

    value = _math.sin(0.5) / 3.0
    expected = np.array([[0.0] * 3, [value] * 3], dtype=np.float32)
    np.testing.assert_allclose(viewer.image, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Group 14 -- a swizzle store that reads the vector it writes
# ---------------------------------------------------------------------------


def test_a_swizzle_store_reads_its_value_before_writing(gpu_backend):
    """``p.xy = p.yx`` is a swap, and a swap with an aliased read is not one.

    CUDA used to emit two component stores, each re-evaluating the value expression:
    ``p.x = make_float2(p.y, p.x).x`` overwrites ``p.x``, and the second store then reads
    the overwritten component, so ``float2(1, 2).xy = .yx`` answers (2, 2) on the device
    and (2, 1) on the interpreter. The value has to land in a temporary first -- python
    evaluates the right-hand side once -- the way every other aliasing-sensitive store in
    the translation already does. Found by the Light Cycle port, whose ``foldSquare``
    swaps the folded coordinates exactly this way; with the bug the tunnel's quadrant fold
    collapses and the corner light streams flood every wall.
    """
    out = np.zeros(2, dtype=np.float64)
    swap_through_swizzle[gpu_backend](out)
    np.testing.assert_allclose(out, [2.0, 1.0])


# ---------------------------------------------------------------------------
# Group 15 -- the receiver struct's layout when its members mix widths
# ---------------------------------------------------------------------------


def test_mixed_width_uniforms_read_their_own_bytes(gpu_backend):
    """float2 + float + int + float4 as instance attributes: each member reads its own.

    The synthesized receiver struct used `ctypes` alignment (a vector member at any
    4-byte offset) while the compiler laid the same declaration out by the device rules
    (`float4` is 16-byte aligned on CUDA), so every member after the first vector read
    the wrong bytes. Found by the promo renderer's uniform set. The struct is now placed
    with the device's own table and explicit padding, and the rendered declaration
    carries the same padding, so the packed bytes are the device's structure.
    """
    viewer = ViewerLike()
    out = np.zeros(4, dtype=np.float32)
    viewer.read[gpu_backend](out)
    np.testing.assert_allclose(out, [2000.0, 4.0, 7.0, 10.0], rtol=1e-5)


def test_class_and_instance_attributes_bind_in_one_kernel(gpu_backend):
    """Both attribute kinds read by one kernel answer their own values."""
    probe = ClassAndInstance()
    out = np.zeros(2, dtype=np.float32)
    probe.read[gpu_backend](out)
    np.testing.assert_allclose(out, [153.0, 18.0], rtol=1e-5)


def test_a_narrower_member_before_a_wider_one_keeps_its_offset(gpu_backend):
    """The int member between two float-width members is the alignment tripwire."""
    probe = Float2BeforeInt()
    out = np.zeros(1, dtype=np.float32)
    probe.read[gpu_backend](out)
    np.testing.assert_allclose(out, [6.0 + 8.0 + 5 + 1.5], rtol=1e-5)
