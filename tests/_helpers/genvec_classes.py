"""The class matrix and helpers the genVec-unification specs share.

Every spec module runs the same questions over a representative class from each
dtype family and width: pygf's float/int/uint/double 2/3/4, PyGPGPU's
char/uchar/short/ushort families, and the 8/16 widths -- which are OpenCL
storage types living beside that backend
(`pygpgpu.opencl.driver.vec_types`), not kernel types: a kernel refuses the
width everywhere.
"""
from __future__ import annotations

from pygf import double2, double3, float2, float3, float4

from pygpgpu.vec_types import (
    char2, char4,
    uchar2, uchar4,
    short2, short4,
    ushort2, ushort4,
)
from pygpgpu.opencl.driver.vec_types import (
    char8, char16,
    int8, int16, uint8, uint16,
    float8, float16,
    double8, double16,
)

# A width's numeric charset: OpenCL spells components s0..sN-1, and a 16-wide
# vector names the slots above nine in hex, either case.
NUMERIC_CHARSET = {8: "01234567", 16: "0123456789ABCDEFabcdef"}


NARROW_FLOAT_CLASSES = [float2, float3, float4, double2, double3]
NARROW_CLASSES = [char2, char4, uchar2, uchar4, short2, short4, ushort2, ushort4]
WIDE_CLASSES = [char8, char16, int8, int16, uint8, uint16,
                float8, float16, double8, double16]
ALL_OWN_CLASSES = NARROW_CLASSES + WIDE_CLASSES + NARROW_FLOAT_CLASSES


def width_of(cls: type) -> int:
    return len(cls())


def family_base(cls: type) -> str:
    """`char16` -> `char`: the family name the class's own result types share."""
    name: str = cls.__name__
    return name.rstrip("0123456789")
