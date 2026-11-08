# The fast approximations each backend spells in its own way.
from __future__ import annotations
from .guards import _scalar_only
import math



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
