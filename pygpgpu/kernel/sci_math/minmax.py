# The ordering pair and the complex-aware minimum and maximum.
from __future__ import annotations
from .guards import _is_complex, _scalar_only
import math





def _complex_less(a:complex, b:complex)->bool:
    """numpy's order on a complex value: the real parts decide, the imaginary ones break a tie.

    `np.minimum`/`np.maximum` and their device implementation both use it, and so does the
    builtin `min`/`max` a kernel body may write (`pygf`'s, on the host).
    """
    return (a.real < b.real) or (a.real == b.real and a.imag < b.imag)


def _complex_fminmax(x:object, y:object, larger:bool)->complex:
    """`fmin`/`fmax` for a complex pair, with numpy's NaN-ignoring rule.

    numpy has two min/max families and they differ on NaN, which is the whole reason both
    exist: `minimum`/`maximum` propagate a NaN (it has a defined place in the order), while
    `fmin`/`fmax` return the operand that is *not* NaN. A complex value is NaN when either
    part is, so that test is made per operand and the surviving one is then ordered.
    """
    left:complex = complex(x)
    right:complex = complex(y)
    left_nan:bool = left.real != left.real or left.imag != left.imag
    right_nan:bool = right.real != right.real or right.imag != right.imag
    if left_nan and right_nan:
        return left
    if left_nan:
        return right
    if right_nan:
        return left
    if left.real == right.real and left.imag == right.imag:
        return left
    return left if (_complex_less(left, right) != larger) else right


def fmin(x:float, y:float)->float:
    """The smaller of two numbers, ignoring a NaN operand (numpy's `fmin`)."""
    _scalar_only("fmin", x)
    _scalar_only("fmin", y)
    if _is_complex(x) or _is_complex(y):
        return _complex_fminmax(x, y, larger=False)
    return min(x, y)


def fmax(x:float, y:float)->float:
    """The larger of two numbers, ignoring a NaN operand (numpy's `fmax`)."""
    _scalar_only("fmax", x)
    _scalar_only("fmax", y)
    if _is_complex(x) or _is_complex(y):
        return _complex_fminmax(x, y, larger=True)
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
