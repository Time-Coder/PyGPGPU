# The distribution functions and the trigonometric-with-pi forms.
from __future__ import annotations
from .guards import _scalar_only
import math



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
