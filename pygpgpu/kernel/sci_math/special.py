# The special functions: erf, the gamma family, and the rounding rule.
from __future__ import annotations
from .guards import _is_complex, _scalar_only
import math



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
    """Round to the nearest integer, halfway cases to even (python's `round`).

    A **complex** value rounds one part at a time, which is what the device's
    `c64_rint`/`c128_rint` do and what numpy's `np.rint(z)` answers: the two sides of a
    complex value have no order between them, but each real part rounds on its own.
    Python's `round` itself refuses a complex value, which is why this is spelled out
    here rather than delegating to it.
    """
    _scalar_only("rint", x)
    if _is_complex(x):
        return complex(round(float(x.real)), round(float(x.imag)))
    return float(round(x))


def remainder(x:float, y:float)->float:
    """The IEEE remainder: x - n*y where n is the nearest integer to x/y."""
    _scalar_only("remainder", x)
    _scalar_only("remainder", y)
    return math.remainder(x, y)
