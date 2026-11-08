# The float/integer conversions, in both directions and both widths.
from __future__ import annotations
from .guards import _scalar_only
import math



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
