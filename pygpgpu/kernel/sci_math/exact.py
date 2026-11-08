# The exactly-rounded forms, and the fused multiply-add.
from __future__ import annotations
from .guards import _scalar_only
import math



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
    fused:object = getattr(math, "fma", None)
    return fused(x, y, z) if fused is not None else x * y + z
