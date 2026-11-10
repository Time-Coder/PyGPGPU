"""Known-failure regressions found by rendering a deep-zoom Mandelbrot on the GPU.

Every test here fails today, on purpose: they are the specification for the fixes, written
as failing tests first (TDD). Each one names the repro that found it and what the right
answer is, and none of the answers invents a new rule -- each is what python (or numpy,
where the interpreter runs numpy) already computes for the same source. The kernels live
one module per bug in `known_failures/`, because the generated source of one python module
is one translation unit: a kernel whose bug is a compile error would otherwise take every
other kernel in the module down with it.

The failures fall into four groups:

* **complex numbers**: `+=` on a complex value fails to compile, `z ** 2` goes through the
  general exp-log power, `bool(z)` fails to compile, `complex(re, im)` truncates numpy
  double parts, and an int multiply wraps silently.
* **64-bit arithmetic**: under `float=np.float64` the transcendental functions are still
  emitted with CUDA's float spellings (`sinf`, `log2f`, ...), a body literal is truncated
  to a float32 constant, `np.sqrt(2)` computes in float32 against numpy's float64 answer,
  a device helper cannot return a tuple, and `pow(2, 10)` (int arguments) fails to compile.
* **the scalar cast refusal**: `two = np.float64(2.0)` below the magic loop is refused as
  a "creation", although the same statement above the loop and inside a device helper are
  both accepted -- the dtype-constructor names sit on the host-creation list.
* **python semantics the translator owes**: a variable refused for widening float to
  double, `min`/`max` capped at two arguments, ndindex extents held in locals read as a
  "second magic loop", an ndenumerate index that cannot be subscripted, and a float
  carried scalar accumulated through an int atomic (which truncates every contribution).
* **a subclass override and its helpers**: a method a subclass overrides may not call a
  helper of the base class, nor a module-level function -- the generated source calls it
  without defining it -- and two subclasses of one `@kernel` share the first one's
  compiled binary, so the second silently answers the first one's colours.
"""
from __future__ import annotations

import numpy as np
import pytest

from known_failures.complex_ctor_parts import complex_constructor_double_parts
from known_failures.complex_inplace import complex_inplace_add
from known_failures.complex_square import complex_square_c64
from known_failures.complex_truth import complex_truthiness
from known_failures.double_literal import double_body_literal
from known_failures.double_math import double_math_functions
from known_failures.float_carried import float_carried_sum
from known_failures.helper_tuple import unpacks_two_scalars
from known_failures.int_overflow import (literal_multiply_overflows,
                                         parameter_multiply_wide)
from known_failures.minmax_arity import min_with_three_arguments
from known_failures.ndenumerate_index import ndenumerate_index_subscript
from known_failures.ndindex_locals import ndindex_with_local_bounds
from known_failures.np_int_math import numpy_math_on_ints
from known_failures.pow_integers import builtin_pow_integers
from known_failures.scalar_cast_loop import scalar_cast_inside_the_loop
from known_failures.subclass_calls_base_helper import Base as SubclassBaseHelperBase
from known_failures.subclass_calls_module_helper import Base as SubclassModuleHelperBase
from known_failures.subclass_calls_module_helper import scaled
from known_failures.subclass_kernel_identity import Base as SubclassIdentityBase
from known_failures.widen_to_double import variable_widens_from_float_to_double

# ---------------------------------------------------------------------------
# Group 1 -- complex numbers
# ---------------------------------------------------------------------------


def test_complex_augassign_compiles_and_matches_the_plain_spelling(gpu_backend):
    """`z += c` is `z = z + c`; today it fails with `no operator "+=" (c128 += c128)`.

    `_visit_augassign` (translator.py) has no complex path: the structure branch looks for
    a class `__iadd__`, finds none on the header's record, and the value falls through to
    the C `+=`, which no compiler accepts for a struct. The right emission is the same
    store `_emit_complex_binop` writes for `z = z + c`.
    """
    out = np.zeros(2, dtype=np.float64)
    complex_inplace_add[gpu_backend](np.float64(0.3), np.float64(0.5), out)
    # c = 0.3 + 0.5j; after four additions both spellings hold 1.2 + 2.0j, so the
    # sum of the four parts is 1.2 + 1.2 + 2.0 + 2.0.
    assert out[0] == pytest.approx(6.4)


def test_complex_square_by_operator_matches_python(gpu_backend):
    """`z ** 2` answers what numpy answers, not an exp-log round trip.

    The general complex power computes exp(2 * log(z)). At c64 with z = 4 - 4j that
    round trip answers 2e-15 - 31.999998j where numpy's `z ** 2` answers -32j exactly --
    the real part is pure noise the direct product never produces, and at larger
    magnitudes the error grows (measured worst 1.4e+24 relative over |z| <= 4). The real
    path already special-cases `x ** 2` into `x * x` (translator.py `_visit_binop`, "Pow"
    with a constant 2); the complex path owes the same special case, and this is exactly
    the Mandelbrot iteration shape.
    """
    a = np.array([4 - 4j], dtype=np.complex64)
    out = np.zeros(1, dtype=np.complex64)
    complex_square_c64[gpu_backend](a, out)
    # The direct product answers the exact value; the exp-log path answers a real part
    # of 2e-15. Asserting bit-exact equality on both parts is what the special case buys.
    assert out[0] == a[0] * a[0]


def test_int_arithmetic_does_not_wrap_silently(backend):
    """A literal product that exceeds int32 must not reach the device wrapped.

    The *parameter* is width-checked (`_check_integer_fits` refuses 2**31 + 5), but
    arithmetic between two in-body ints was not, so `5000000 * 1000000` reached the
    device as 658067456 with a compiler warning nobody pipes up. Both sides are
    literals, so the exact value folds at translation time and runs the same check a
    scalar argument gets; the refusal names the numpy spelling that carries the value.
    """
    out = np.zeros(1, dtype=np.float64)
    if backend == "python":
        # The interpreter runs python's own arithmetic and answers correctly already.
        literal_multiply_overflows["python"](out)
        assert out[0] == 5_000_000_000_000
        return
    with pytest.raises((ValueError, SyntaxError, OverflowError)) as caught:
        literal_multiply_overflows[backend](out)
    message = str(caught.value)
    assert "5000000000000" in message or "int64" in message


def test_a_wide_int_mapping_carries_the_parameter_product(gpu_backend):
    """`step * 1000000` under `int=np.int64` answers python's number.

    A launch-time product cannot be checked at translation time (the value is not known),
    and the existing escape hatch is the `int=np.int64` mapping -- which the translator
    already honours by widening the whole arithmetic (`long long big = step * 1000000`).
    This pins the mechanism the refusal above points at.
    """
    out = np.zeros(1, dtype=np.float64)
    parameter_multiply_wide[gpu_backend](5000, out)
    assert out[0] == 5_000_000_000


def test_bool_of_a_complex_value_is_its_non_zeroNESS(gpu_backend):
    """`if z:` is `z != 0` in python (either part non-zero); today it is a compile
    error: "expression must have bool type" -- the record is emitted into the condition
    verbatim.
    """
    out = np.zeros((1, 2), dtype=np.float64)
    complex_truthiness[gpu_backend](out)
    assert out[0, 0] == 0.0
    assert out[0, 1] == 1.0


def test_complex_constructor_keeps_numpy_parts_whole(gpu_backend):
    """`complex(re, im)` with double parts truncates both to float32 today.

    Under `float=np.float64` the two parts are doubles (a python float argument travels
    as the mapping's double), and numpy's own `complex(1.0000001234567890,
    -0.12345678901234567)` is a complex128 -- the constructor should widen to the wider
    part, the same rule every mixed operation follows: nothing is rounded down.
    """
    out = np.zeros((1, 2), dtype=np.float64)
    complex_constructor_double_parts[gpu_backend](
        1.0000001234567890, -0.12345678901234567, out)
    assert out[0, 0] == pytest.approx(1.0000001234567890, rel=1e-15)
    assert out[0, 1] == pytest.approx(-0.12345678901234567, rel=1e-15)


# ---------------------------------------------------------------------------
# Group 2 -- 64-bit arithmetic under float=np.float64
# ---------------------------------------------------------------------------


def test_math_functions_compute_at_the_mapped_width(gpu_backend):
    """`float=np.float64` must give double-precision transcendentals, not sinf(x).

    The generated CUDA source spells every function with the float suffix and casts the
    double argument down: `sinf((float)x)`. The measured error for np.sin is ~2.4e-8 and
    for np.exp ~1.8e-7 -- exactly float32's epsilon. A double kernel's smooth-colour
    arithmetic inherits it and bands appear in deep zooms. The kernel's python float
    argument travels as the mapping's double, so `ret` widens and the plain (double)
    name is what the compiler resolves.
    """
    x = 1.0000001234567890
    out = np.zeros((1, 4), dtype=np.float64)
    double_math_functions[gpu_backend](x, out)
    np.testing.assert_allclose(out[0], [np.sin(x), np.cos(x), np.sqrt(x), np.exp(x)],
                               rtol=1e-12, atol=1e-12)


def test_a_body_literal_keeps_double_digits_under_a_double_mapping(gpu_backend):
    """`0.1234567890123456789` arrives as float32(0.12345679) today.

    The complex literal already follows the mapping (a python complex is c128 under
    float=np.float64), so the real literal beside it should follow it too: python's float
    literal *is* a double, and the mapping is the feature that says "64-bit reals here".
    The default mapping is unaffected -- a python float still travels as float32 there.
    """
    out = np.zeros(2, dtype=np.float64)
    double_body_literal[gpu_backend](out)
    assert out[0] == pytest.approx(0.1234567890123456789, rel=1e-15)


def test_numpy_math_on_int_literals_is_float64_like_numpy(gpu_backend):
    """`np.sqrt(2)` is a numpy float64; the device emits `sqrtf((float)2)`.

    The interpreter runs numpy and answers 1.4142135623730951; the device answers
    1.4142135381698608. numpy's rule for an int *input* is "compute in float64", and the
    translator already has the double spelling (`sqrt`) -- it just never selects it for a
    python int argument.
    """
    out = np.zeros((1, 3), dtype=np.float64)
    numpy_math_on_ints[gpu_backend](out)
    np.testing.assert_allclose(out[0], [np.sqrt(2), np.log(10), np.exp(1)],
                               rtol=1e-12, atol=1e-12)


def test_a_device_helper_can_return_two_scalars(gpu_backend):
    """`return a, b` from a helper, unpacked at the call site.

    `@kernel` itself supports tuple returns (the ReturnPosition machinery), so the helper
    form is the same feature at a smaller scale -- but today the helper body is refused
    with "unsupported expression: Tuple" and the call site with "tuple assignment
    requires matching tuples".
    """
    params = np.array([3.0, 1.0], dtype=np.float64)
    out = np.zeros((1, 2), dtype=np.float64)
    unpacks_two_scalars[gpu_backend](params, out)
    assert out[0, 0] == 4.0
    assert out[0, 1] == 2.0


def test_the_python_pow_builtin_takes_integer_arguments(runner):
    """`pow(2, 10)` is 1024; today the device emits `pow((int)2, (int)10)`, which CUDA's
    NVRTC and OpenCL's clang refuse as ambiguous ("more than one instance of overloaded
    function pow"). HIP's clang accepts it by accident -- its `<cmath>` resolves the int
    overload -- so the same source answers on one backend and refuses on the others.

    python's builtin `pow` with two ints answers an int; the device spelling should cast
    to float the way `**` already does, and answer 1024 on every backend.
    """
    out = np.zeros(1, dtype=np.float64)
    runner(builtin_pow_integers, out)
    assert float(out[0]) == 1024.0


# ---------------------------------------------------------------------------
# Group 3 -- the scalar cast refused as a "creation"
# ---------------------------------------------------------------------------


def test_a_scalar_dtype_cast_below_the_magic_loop_is_not_a_creation(gpu_backend):
    """`np.float64(2.0)` inside the loop is refused today as a per-work-item creation.

    The dtype-constructor names (float64, complex64, ...) sit on _HOST_CREATION_FUNCTIONS,
    so a scalar cast trips the creation guard. The same statement above the loop is lifted
    to the host and inside a device helper it translates as a cast -- three answers for one
    spelling. A cast is a cast wherever it appears.
    """
    out = np.zeros(2, dtype=np.float64)
    scalar_cast_inside_the_loop[gpu_backend](np.float64(0.9), out)
    z = 0.9
    for _ in range(3):
        z = z * z
    # The parameter travels as a float (the documented narrowing), so the comparison
    # carries float32's rounding; the point of the test is that the cast spelling
    # compiles and answers the folded value, not the parameter's width.
    assert out[0] == pytest.approx(z * 2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Group 4 -- python semantics the translator owes
# ---------------------------------------------------------------------------


def test_a_variable_widens_from_float_to_double(gpu_backend):
    """A name first bound to a python float may later hold a wider value.

    Python's variables are untyped, and the translator already widens int -> float
    (`_define` accepts that direction); float -> double is the same widening one step
    further. Today it refuses with "variable 'scale' changes type from float to double",
    which forces every double kernel to pre-declare its accumulators as np.float64 above
    the loop -- and that spelling is refused inside the loop (the group-3 test).
    """
    out = np.zeros(2, dtype=np.float64)
    variable_widens_from_float_to_double[gpu_backend](np.float64(0.25), out)
    assert out[0] == 0.5


def test_min_and_max_take_three_arguments(runner):
    """`min(x, 0.7, 0.9)` is refused today with "takes exactly 2 arguments".

    python chains the comparisons (`min(min(a, b), c)`), numpy has no three-argument
    minimum, and a device has no reason to refuse the spelling.
    """
    out = np.zeros(1, dtype=np.float32)
    runner(min_with_three_arguments, np.float32(0.5), out)
    assert float(out[0]) == pytest.approx(min(0.5, 0.7, 0.9) + max(0.5, 0.2, 0.1))


def test_ndindex_accepts_extents_held_in_locals(runner):
    """`rows, columns = ...` bounds must not make the loop stop being the magic loop.

    Today the loop is not recognized as the dispatch and the body is refused with "this
    kernel body has more than one magic loop" -- although the source contains exactly one
    loop. The recognition should follow the data (the bounds are `out.shape` members), or
    the refusal should say "the loop bounds must be arr.shape", not count loops.
    """
    out = np.zeros((3, 4), dtype=np.float64)
    runner(ndindex_with_local_bounds, out)
    np.testing.assert_allclose(out, np.arange(12, dtype=np.float64).reshape(3, 4))


def test_ndenumerate_index_can_be_subscripted(runner):
    """`index[0]` on an ndenumerate index is refused today ("only array ... can be
    indexed"), although python hands the index over as a tuple and `out[index]` works.
    """
    out = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    runner(ndenumerate_index_subscript, out)
    np.testing.assert_allclose(out, [2.0, 4.0, 6.0])


def test_a_float_carried_scalar_accumulates_in_its_own_width(runner):
    """`total = 0.0` above the loop, `total += x[i]` inside it -- the sum loses today.

    The carried-scalar machinery declares the shared slot `int*` for every type and emits
    `atomicAdd(total, x[i])`: an int atomic handed a double value. The contributions are
    truncated (`0.5 + 0.25` answers 0), and the interpreter answers python's own 0.75, so
    the same source disagrees between backends. The slot has to be the carried value's own
    width (a float atomic, or a per-work-item private plus one deterministic reduction).
    """
    x = np.array([0.5, 0.25], dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    runner(float_carried_sum, x, out)
    assert out[0] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Group 5 -- a subclass override and the helpers it calls
#
# Found by writing `examples/fractal/fractal_viewer.py`: a base class owns the @kernel and
# its body calls an overridable hook, and three viewers in other modules write that hook.
# Two of the three compile; the one whose override reaches a helper of the base does not.
# ---------------------------------------------------------------------------


def _sample_grid(factor: float) -> np.ndarray:
    """What `sample(i, j)` answers when it is `factor * (i + j + 1)`."""
    rows = np.arange(2, dtype=np.float32)[:, None]
    columns = np.arange(3, dtype=np.float32)[None, :]
    return factor * (rows + columns + 1.0)


class _UsesInheritedHelper(SubclassBaseHelperBase):
    """The override colours through a helper it inherits, the way the examples do."""

    def sample(self, i: int, j: int) -> np.float32:
        return self.scale(np.float32(i + j + 1))


def test_a_subclass_override_can_call_an_inherited_helper(gpu_backend):
    """`self.scale(...)` from an override must compile, exactly as from the kernel body.

    The generated source calls the base's device function and never defines it, so the module
    is refused with `identifier "Base_scale_<hash>" is undefined`. Nothing about the call is
    ambiguous -- the receiver is the instance, the class is known at translation time -- and
    the same helper is emitted correctly when the *kernel body* calls it directly, so the walk
    that collects device functions simply has to reach through an override too. The expected
    numbers are python's: `sample(i, j)` is `2 * (i + j + 1)`.
    """
    viewer = _UsesInheritedHelper()
    viewer.draw[gpu_backend]()
    np.testing.assert_allclose(viewer.image, _sample_grid(2.0))


class _UsesModuleHelper(SubclassModuleHelperBase):
    """The override colours through a module-level function."""

    def sample(self, i: int, j: int) -> np.float32:
        return scaled(np.float32(i + j + 1))


def test_a_subclass_override_can_call_a_module_level_function(gpu_backend):
    """A module-level device helper is reachable from a kernel body and from an override alike.

    Same failure as the inherited helper above (`identifier "scaled_<hash>" is undefined`), and
    the same reason: the helpers an override reaches are not walked for emission. The expected
    numbers are python's: `sample(i, j)` is `3 * (i + j + 1)`.
    """
    viewer = _UsesModuleHelper()
    viewer.draw[gpu_backend]()
    np.testing.assert_allclose(viewer.image, _sample_grid(3.0))


class _AnswersOne(SubclassIdentityBase):
    """The first subclass to launch."""

    def sample(self, i: int, j: int) -> np.float32:
        return np.float32(1)


class _AnswersTwo(SubclassIdentityBase):
    """A second subclass, whose hook answers differently."""

    def sample(self, i: int, j: int) -> np.float32:
        return np.float32(2)


def test_two_subclasses_of_one_kernel_keep_their_own_hook(gpu_backend):
    """The second subclass must not be served the first one's device code.

    The compiled-launch cache is keyed by backend, device, the *argument* signature and the
    type map; the class whose hook the kernel inlined is in none of them, so the second launch
    reuses the first binary and every pixel comes back 1 instead of 2. That is the worst kind
    of failure -- a wrong answer with no error -- and it is what a script showing two viewers
    of one base class hits. Whatever supplies the identity, two different hooks must not share
    one cached launch.
    """
    first = _AnswersOne()
    second = _AnswersTwo()
    first.draw[gpu_backend]()
    second.draw[gpu_backend]()
    np.testing.assert_allclose(first.image, 1.0)
    np.testing.assert_allclose(second.image, 2.0)
