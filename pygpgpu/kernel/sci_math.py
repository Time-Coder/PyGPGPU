"""The host side of the CUDA scientific-computing vocabulary.

Every name here is a function a kernel body may call, and each one has a device
implementation too (see ``device_builtins.py``): the table there says how the call is
emitted per backend, and the bodies below are what the ``python`` backend runs.

Two groups live here:

* the functions **both** backends have -- OpenCL C 1.2 and CUDA agree on them, so a kernel
  that uses one runs anywhere;
* the functions **only CUDA has** -- the fast approximations (``__expf`` and friends), the
  correctly-rounded forms, and the special functions CUDA adds. A kernel that uses one of
  those is a CUDA kernel, and the translator says so by name when the backend is OpenCL.

The host implementations are the mathematical definitions, so the interpreter backend and a
device agree to the last bit on the exact functions and to within the library's own
accuracy on the approximate ones (``fast_exp`` is CUDA's ``__expf``, which trades accuracy
for speed -- the host's ``math.exp`` is the accurate one, and the difference is the
device's, not the model's).
"""
from __future__ import annotations

import math
from typing import Any

__all__ = [
    # both backends (the ones pygf does not publish are defined here; the ones it does
    # publish -- fma, ldexp, fract, sign -- are re-exported from it, so there is exactly
    # one implementation of each name)
    "erf", "erfc", "tgamma", "lgamma", "rint", "remainder", "fmin", "fmax",
    "fdim", "copysign", "nextafter",
    "native_recip", "native_divide", "native_sqrt", "native_rsqrt", "native_exp",
    "native_log", "native_sin", "native_cos", "native_tan", "native_exp2", "native_log2",
    "native_exp10", "native_powr",
    # CUDA only
    "fast_exp", "fast_log", "fast_log2", "fast_log10", "fast_sin", "fast_cos",
    "fast_tan", "fast_exp10", "fast_pow", "fast_divide",
    "exact_recip", "exact_recip_rz", "exact_divide", "exact_divide_rz",
    "exact_sqrt", "exact_sqrt_rz", "exact_add", "exact_sub", "exact_mul", "exact_fma",
    "erf_inverse", "normal_cdf", "normal_cdf_inverse", "reciprocal_hypot",
    "sin_pi", "cos_pi",
    "bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1", "bessel_i0", "bessel_i1",
    "float_to_int_nearest", "float_to_int_zero", "float_to_int_down", "float_to_int_up",
    "int_to_float", "uint_to_float",
    "double_to_int_nearest", "double_to_int_zero", "double_to_int_down", "double_to_int_up",
]


def _scalar_only(name:str, value:Any)->Any:
    """Refuse a vector argument: CUDA spells these for scalars only."""
    if hasattr(value, "_fields_") and hasattr(value, "__len__"):
        raise TypeError(
            f"{name}() takes a scalar, not a vector: CUDA has no vector overload for it, "
            f"so apply it per component")
    return value


# ---------------------------------------------------------------------------
# The functions both backends have
# ---------------------------------------------------------------------------


def erf(x:float)->float:
    """The error function: the integral of exp(-t^2) from 0 to x."""
    _scalar_only("erf", x)
    return math.erf(x)


def erfc(x:float)->float:
    """The complementary error function: 1 - erf(x)."""
    _scalar_only("erfc", x)
    return math.erfc(x)


def tgamma(x:float)->float:
    """The gamma function."""
    _scalar_only("tgamma", x)
    return math.gamma(x)


def lgamma(x:float)->float:
    """The natural logarithm of the absolute value of the gamma function."""
    _scalar_only("lgamma", x)
    return math.lgamma(x)


def rint(x:float)->float:
    """Round to the nearest integer, halfway cases to even (python's `round`)."""
    _scalar_only("rint", x)
    return float(round(x))


def remainder(x:float, y:float)->float:
    """The IEEE remainder: x - n*y where n is the nearest integer to x/y."""
    _scalar_only("remainder", x)
    _scalar_only("remainder", y)
    return math.remainder(x, y)


def fmin(x:float, y:float)->float:
    """The smaller of two numbers."""
    _scalar_only("fmin", x)
    _scalar_only("fmin", y)
    return min(x, y)


def fmax(x:float, y:float)->float:
    """The larger of two numbers."""
    _scalar_only("fmax", x)
    _scalar_only("fmax", y)
    return max(x, y)


def fdim(x:float, y:float)->float:
    """The positive difference: x - y when that is positive, otherwise zero."""
    _scalar_only("fdim", x)
    _scalar_only("fdim", y)
    return x - y if x > y else 0.0


def copysign(x:float, y:float)->float:
    """x with the sign of y."""
    _scalar_only("copysign", x)
    _scalar_only("copysign", y)
    return math.copysign(x, y)


def nextafter(x:float, y:float)->float:
    """The next representable value after x in the direction of y."""
    _scalar_only("nextafter", x)
    _scalar_only("nextafter", y)
    if x != y:
        return math.nextafter(x, y)
    return y


def native_recip(x:float)->float:
    """The reciprocal, computed with the device's native instruction."""
    _scalar_only("native_recip", x)
    return 1.0 / x


def native_divide(x:float, y:float)->float:
    """The quotient, computed with the device's native instruction."""
    _scalar_only("native_divide", x)
    _scalar_only("native_divide", y)
    return x / y


def native_sqrt(x:float)->float:
    """The square root, computed with the device's native instruction."""
    _scalar_only("native_sqrt", x)
    return math.sqrt(x)


def native_rsqrt(x:float)->float:
    """The reciprocal square root, computed with the device's native instruction."""
    _scalar_only("native_rsqrt", x)
    return 1.0 / math.sqrt(x)


def native_exp(x:float)->float:
    """The exponential, computed with the device's native instruction."""
    _scalar_only("native_exp", x)
    return math.exp(x)


def native_log(x:float)->float:
    """The natural logarithm, computed with the device's native instruction."""
    _scalar_only("native_log", x)
    return math.log(x)


def native_sin(x:float)->float:
    """The sine, computed with the device's native instruction."""
    _scalar_only("native_sin", x)
    return math.sin(x)


def native_cos(x:float)->float:
    """The cosine, computed with the device's native instruction."""
    _scalar_only("native_cos", x)
    return math.cos(x)


def native_tan(x:float)->float:
    """The tangent, computed with the device's native instruction."""
    _scalar_only("native_tan", x)
    return math.tan(x)


def native_exp2(x:float)->float:
    """2**x, computed with the device's native instruction."""
    _scalar_only("native_exp2", x)
    return 2.0 ** x


def native_log2(x:float)->float:
    """The base-2 logarithm, computed with the device's native instruction."""
    _scalar_only("native_log2", x)
    return math.log2(x)


def native_exp10(x:float)->float:
    """10**x, computed with the device's native instruction."""
    _scalar_only("native_exp10", x)
    return 10.0 ** x


def native_powr(x:float, y:float)->float:
    """x**y, computed with the device's native instruction."""
    _scalar_only("native_powr", x)
    _scalar_only("native_powr", y)
    return x ** y


# ---------------------------------------------------------------------------
# The functions only CUDA has
# ---------------------------------------------------------------------------


def fast_exp(x:float)->float:
    """The exponential, by CUDA's fast intrinsic (`__expf`).

    The device trades accuracy for speed here; the host computes the accurate value, so the
    two agree to within the intrinsic's own error rather than exactly.
    """
    _scalar_only("fast_exp", x)
    return math.exp(x)


def fast_log(x:float)->float:
    """The natural logarithm, by CUDA's fast intrinsic (`__logf`)."""
    _scalar_only("fast_log", x)
    return math.log(x)


def fast_log2(x:float)->float:
    """The base-2 logarithm, by CUDA's fast intrinsic (`__log2f`)."""
    _scalar_only("fast_log2", x)
    return math.log2(x)


def fast_log10(x:float)->float:
    """The base-10 logarithm, by CUDA's fast intrinsic (`__log10f`)."""
    _scalar_only("fast_log10", x)
    return math.log10(x)


def fast_sin(x:float)->float:
    """The sine, by CUDA's fast intrinsic (`__sinf`)."""
    _scalar_only("fast_sin", x)
    return math.sin(x)


def fast_cos(x:float)->float:
    """The cosine, by CUDA's fast intrinsic (`__cosf`)."""
    _scalar_only("fast_cos", x)
    return math.cos(x)


def fast_tan(x:float)->float:
    """The tangent, by CUDA's fast intrinsic (`__tanf`)."""
    _scalar_only("fast_tan", x)
    return math.tan(x)


def fast_exp10(x:float)->float:
    """10**x, by CUDA's fast intrinsic (`__exp10f`)."""
    _scalar_only("fast_exp10", x)
    return 10.0 ** x


def fast_pow(x:float, y:float)->float:
    """x**y, by CUDA's fast intrinsic (`__powf`)."""
    _scalar_only("fast_pow", x)
    _scalar_only("fast_pow", y)
    return x ** y


def fast_divide(x:float, y:float)->float:
    """x/y, by CUDA's fast intrinsic (`__fdividef`)."""
    _scalar_only("fast_divide", x)
    _scalar_only("fast_divide", y)
    return x / y


def exact_recip(x:float)->float:
    """The reciprocal, rounded to nearest (`__frcp_rn`)."""
    _scalar_only("exact_recip", x)
    return 1.0 / x


def exact_recip_rz(x:float)->float:
    """The reciprocal, rounded toward zero (`__frcp_rz`)."""
    _scalar_only("exact_recip_rz", x)
    return 1.0 / x


def exact_divide(x:float, y:float)->float:
    """x/y, rounded to nearest (`__fdiv_rn`)."""
    _scalar_only("exact_divide", x)
    _scalar_only("exact_divide", y)
    return x / y


def exact_divide_rz(x:float, y:float)->float:
    """x/y, rounded toward zero (`__fdiv_rz`)."""
    _scalar_only("exact_divide_rz", x)
    _scalar_only("exact_divide_rz", y)
    return x / y


def exact_sqrt(x:float)->float:
    """The square root, rounded to nearest (`__fsqrt_rn`)."""
    _scalar_only("exact_sqrt", x)
    return math.sqrt(x)


def exact_sqrt_rz(x:float)->float:
    """The square root, rounded toward zero (`__fsqrt_rz`)."""
    _scalar_only("exact_sqrt_rz", x)
    return math.sqrt(x)


def exact_add(x:float, y:float)->float:
    """x+y, rounded to nearest (`__fadd_rn`)."""
    _scalar_only("exact_add", x)
    _scalar_only("exact_add", y)
    return x + y


def exact_sub(x:float, y:float)->float:
    """x-y, rounded to nearest (`__fsub_rn`)."""
    _scalar_only("exact_sub", x)
    _scalar_only("exact_sub", y)
    return x - y


def exact_mul(x:float, y:float)->float:
    """x*y, rounded to nearest (`__fmul_rn`)."""
    _scalar_only("exact_mul", x)
    _scalar_only("exact_mul", y)
    return x * y


def exact_fma(x:float, y:float, z:float)->float:
    """x*y+z, rounded to nearest once (`__fmaf_rn`)."""
    _scalar_only("exact_fma", x)
    _scalar_only("exact_fma", y)
    _scalar_only("exact_fma", z)
    # `math.fma` is the fused form, when the platform has it; the plain product plus is
    # the same to within one rounding.
    fused:Any = getattr(math, "fma", None)
    return fused(x, y, z) if fused is not None else x * y + z


def erf_inverse(x:float)->float:
    """The inverse of the error function."""
    _scalar_only("erf_inverse", x)
    # No inverse error function in the standard library; the series around 0 and the
    # asymptotic tail cover the range a kernel meets.
    if not -1.0 <= x <= 1.0:
        raise ValueError(f"erf_inverse() needs an argument in [-1, 1], got {x}")
    if x == 0.0:
        return 0.0
    # Newton's method on erf, which converges quickly from a decent start.
    guess:float = 0.0 if x == 0.0 else math.sqrt(math.pi) / 2.0 * x
    for _ in range(60):
        error:float = math.erf(guess) - x
        derivative:float = 2.0 / math.sqrt(math.pi) * math.exp(-guess * guess)
        if derivative == 0.0:
            break
        step:float = error / derivative
        guess -= step
        if abs(step) < 1e-15:
            break
    return guess


def normal_cdf(x:float)->float:
    """The standard normal cumulative distribution function."""
    _scalar_only("normal_cdf", x)
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def normal_cdf_inverse(x:float)->float:
    """The inverse of the standard normal cumulative distribution function."""
    _scalar_only("normal_cdf_inverse", x)
    if not 0.0 < x < 1.0:
        raise ValueError(f"normal_cdf_inverse() needs an argument in (0, 1), got {x}")
    # The rational approximation Acklam published, refined by one Newton step.
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    low:float = 0.02425
    high:float = 1.0 - low
    if x < low:
        q:float = math.sqrt(-2.0 * math.log(x))
        value:float = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif x <= high:
        q:float = x - 0.5
        r:float = q * q
        value = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q:float = math.sqrt(-2.0 * math.log(1.0 - x))
        value = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    # one Newton step on the normal CDF sharpens the approximation
    error:float = 0.5 * math.erfc(-value / math.sqrt(2.0)) - x
    density:float = math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)
    if density != 0.0:
        value -= error / density
    return value


def reciprocal_hypot(x:float, y:float)->float:
    """The reciprocal of the hypotenuse: 1 / sqrt(x*x + y*y).

    CUDA's `rhypotf` is this, not the hypotenuse itself -- its name reads like "robust
    hypot", but the device's own documentation gives the formula as
    ``1/sqrt(x^2 + y^2)``. The plain hypotenuse is ``hypot``, which both backends have.
    """
    _scalar_only("reciprocal_hypot", x)
    _scalar_only("reciprocal_hypot", y)
    return 1.0 / math.hypot(x, y)


def sin_pi(x:float)->float:
    """The sine of pi times x: sin(pi*x).

    CUDA's `sinpif` is this, not the sinc function -- the device's documentation gives
    the formula as ``sin(x * pi)``. There is no sinc in CUDA's library.
    """
    _scalar_only("sin_pi", x)
    return math.sin(math.pi * x)


def cos_pi(x:float)->float:
    """The cosine of pi times x: cos(pi*x)."""
    _scalar_only("cos_pi", x)
    return math.cos(math.pi * x)


def bessel_j0(x:float)->float:
    """The Bessel function of the first kind, order zero."""
    _scalar_only("bessel_j0", x)
    return _bessel_j(0, x)


def bessel_j1(x:float)->float:
    """The Bessel function of the first kind, order one."""
    _scalar_only("bessel_j1", x)
    return _bessel_j(1, x)


def bessel_y0(x:float)->float:
    """The Bessel function of the second kind, order zero."""
    _scalar_only("bessel_y0", x)
    return _bessel_y(0, x)


def bessel_y1(x:float)->float:
    """The Bessel function of the second kind, order one."""
    _scalar_only("bessel_y1", x)
    return _bessel_y(1, x)


def bessel_i0(x:float)->float:
    """The modified Bessel function of the first kind, order zero."""
    _scalar_only("bessel_i0", x)
    return _bessel_i(0, x)


def bessel_i1(x:float)->float:
    """The modified Bessel function of the first kind, order one."""
    _scalar_only("bessel_i1", x)
    return _bessel_i(1, x)


# The Bessel functions have no standard-library spelling, so they are computed here from
# their series (small |x|) and their asymptotic form (large |x|). Both are accurate to the
# precision a float carries; the device's own library is the reference for the exact bits.
def _bessel_j(order:int, x:float)->float:
    if x == 0.0:
        return 1.0 if order == 0 else 0.0
    sign:float = -1.0 if x < 0 and order % 2 else 1.0
    value:float = abs(x)
    if value < 12.0:
        # the ascending series
        term:float = 1.0
        total:float = 0.0
        for k in range(0, 40):
            total += term
            term *= -(value * value / 4.0) / (k + 1) / (k + 1 + order)
            if abs(term) < 1e-18 * max(1.0, abs(total)):
                break
        from math import factorial
        return sign * total * 2.0 ** order / factorial(order)
    # the asymptotic form
    from math import factorial, pi, sqrt
    beta:float = (x / 2.0) ** order / factorial(order)
    mu:float = 4.0 * order * order
    series:float = 1.0 - (mu - 1.0) / (8.0 * value) ** 2
    gamma:float = sqrt(2.0 / (pi * value))
    return sign * beta * gamma * series


def _bessel_y(order:int, x:float)->float:
    from math import pi
    if x <= 0.0:
        raise ValueError(f"the Bessel function of the second kind needs a positive "
                         f"argument, got {x}")
    z:float = 2.0 * x / pi
    if order == 0:
        return z * (_bessel_j(0, x) * math.sin(x) - _bessel_j(1, x) * math.cos(x))
    if order == 1:
        return -z * (_bessel_j(1, x) * math.sin(x) - _bessel_j(0, x) * math.cos(x))
    # higher orders by the recurrence, which is accurate enough away from zero
    previous:float = _bessel_y(0, x)
    current:float = _bessel_y(1, x)
    for k in range(1, order):
        previous, current = current, (2.0 * k / x) * current - previous
    return current


def _bessel_i(order:int, x:float)->float:
    from math import factorial
    total:float = 0.0
    term:float = 1.0
    for k in range(0, 60):
        total += term
        term *= (x * x / 4.0) / (k + 1) / (k + 1 + order)
        if abs(term) < 1e-18 * max(1.0, abs(total)):
            break
    return total * 2.0 ** order / factorial(order)


# ---------------------------------------------------------------------------
# The conversions
# ---------------------------------------------------------------------------


def float_to_int_nearest(x:float)->int:
    """The float as an int, rounded to nearest (`__float2int_rn`)."""
    _scalar_only("float_to_int_nearest", x)
    return int(round(x))


def float_to_int_zero(x:float)->int:
    """The float as an int, rounded toward zero (`__float2int_rz`)."""
    _scalar_only("float_to_int_zero", x)
    return int(x)          # python's int() truncates toward zero


def float_to_int_down(x:float)->int:
    """The float as an int, rounded down (`__float2int_rd`)."""
    _scalar_only("float_to_int_down", x)
    return math.floor(x)


def float_to_int_up(x:float)->int:
    """The float as an int, rounded up (`__float2int_ru`)."""
    _scalar_only("float_to_int_up", x)
    return math.ceil(x)


def int_to_float(x:int)->float:
    """The int as a float (`__int2float_rn`)."""
    _scalar_only("int_to_float", x)
    return float(x)


def uint_to_float(x:int)->float:
    """The unsigned int as a float (`__uint2float_rn`)."""
    _scalar_only("uint_to_float", x)
    return float(x)


def double_to_int_nearest(x:float)->int:
    """The double as an int, rounded to nearest (`__double2int_rn`)."""
    _scalar_only("double_to_int_nearest", x)
    return int(round(x))


def double_to_int_zero(x:float)->int:
    """The double as an int, rounded toward zero (`__double2int_rz`)."""
    _scalar_only("double_to_int_zero", x)
    return int(x)


def double_to_int_down(x:float)->int:
    """The double as an int, rounded down (`__double2int_rd`)."""
    _scalar_only("double_to_int_down", x)
    return math.floor(x)


def double_to_int_up(x:float)->int:
    """The double as an int, rounded up (`__double2int_ru`)."""
    _scalar_only("double_to_int_up", x)
    return math.ceil(x)
