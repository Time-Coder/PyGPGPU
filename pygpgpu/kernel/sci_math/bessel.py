# The Bessel functions, in their three families.
from __future__ import annotations
from .guards import _scalar_only
import math



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
