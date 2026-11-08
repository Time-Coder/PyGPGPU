# The fast approximations, one family per backend.
from __future__ import annotations
from .guards import _scalar_only
import math



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
