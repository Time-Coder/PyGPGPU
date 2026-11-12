"""The vector classes the OpenCL driver declares.

Two kinds of class live here: the 14 whose layout differs from the shared one
(every 3-component type -- OpenCL pads to 16 bytes where CUDA/HIP stay compact
-- plus `long2`/`long4`/`ulong2`/`ulong4`), and the 8/16-wide families, which
only OpenCL C declares at all. The 8/16 classes carry the numeric `s0`-style
swizzles through the width bases (`genVec8`/`genVec16`, subclasses of pygf's
`genVec`); a kernel refuses the width everywhere, because a type only one
backend has would make one body mean different things per device.
"""
from .char3 import char3
from .uchar3 import uchar3
from .short3 import short3
from .ushort3 import ushort3
from .int3 import int3
from .uint3 import uint3
from .long3 import long3
from .ulong3 import ulong3
from .float3 import float3
from .double3 import double3
from .long2 import long2
from .ulong2 import ulong2
from .long4 import long4
from .ulong4 import ulong4

from .char8 import char8
from .char16 import char16
from .uchar8 import uchar8
from .uchar16 import uchar16
from .short8 import short8
from .short16 import short16
from .ushort8 import ushort8
from .ushort16 import ushort16
from .int8 import int8
from .int16 import int16
from .uint8 import uint8
from .uint16 import uint16
from .long8 import long8
from .long16 import long16
from .ulong8 import ulong8
from .ulong16 import ulong16
from .float8 import float8
from .float16 import float16
from .double8 import double8
from .double16 import double16

from .genVec8 import genVec8
from .genVec16 import genVec16

# pygf owns the numpy patches and the `VectorArray` view they produce: an array
# of a registered vector type answers `a[i].s0` instead of a plain record. The
# shared classes are registered by `pygpgpu.vec_types`; these are this
# package's own, so they register here. (The registration block sits before the
# re-exports below because it needs only the classes defined above; the E402
# that would normally flag a late import is the point -- the shared 2/4-wide
# classes must be importable whether or not the registration ran first.)
import pygf as _pygf

for _value in (char8, char16, uchar8, uchar16,
               short8, short16, ushort8, ushort16,
               int8, int16, uint8, uint16,
               long8, long16, ulong8, ulong16,
               float8, float16, double8, double16):
    _pygf.register_vector_type(_value)

del _value, _pygf

# The 2/4-wide classes the three backends agree on remain the shared ones
# (pygf's, re-exported through `pygpgpu.vec_types`). A kernel refuses the
# 8/16-wide widths everywhere -- see translator.vectors.KERNEL_VECTOR_WIDTHS.
from ....vec_types import (  # noqa: E402  (deliberately after the registration)
    char2, char4,
    uchar2, uchar4,
    short2, short4,
    ushort2, ushort4,
    int2, int4,
    uint2, uint4,
    float2, float4,
    double2, double4,
)
