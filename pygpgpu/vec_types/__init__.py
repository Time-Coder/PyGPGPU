from .char2 import char2
from .char4 import char4

from .uchar2 import uchar2
from .uchar4 import uchar4

from .short2 import short2
from .short4 import short4

from .ushort2 import ushort2
from .ushort4 import ushort4

# The float/int/uint/double 2 and 4 widths that pygf already provides are
# re-exported from pygf so both libraries share the same classes (identical
# layout and structured dtype). The widths pygf has no counterpart for (8/16)
# and the char/uchar/short/ushort families stay PyGPGPU's own.
from pygf.double2 import double2
from pygf.double3 import double3
from pygf.double4 import double4
from pygf.float2 import float2
from pygf.float3 import float3
from pygf.float4 import float4
from pygf.int2 import int2
from pygf.int3 import int3
from pygf.int4 import int4
from pygf.uint2 import uint2
from pygf.uint3 import uint3
from pygf.uint4 import uint4

# The base classes are pygf's: one vector implementation serves both libraries.
# The width bases below are PyGPGPU's thin subclasses that turn the OpenCL
# numeric swizzles on; genVec itself is re-exported unchanged. The 8/16-wide
# bases live beside the only backend that declares those widths
# (`pygpgpu.opencl.driver.vec_types`): a kernel refuses the width everywhere,
# and a host user of the numeric swizzles imports from there.
from pygf.genVec import genVec
from pygf.genVec2 import genVec2
from pygf.genVec3 import genVec3
from pygf.genVec4 import genVec4

# pygf owns the numpy patches and the `VectorArray` view they produce (see
# pygf/vector_array.py). It knows the types it defines; the char/uchar/short/ushort
# families PyGPGPU adds have to be handed over, or an array of them would come back
# as plain records and `a[i].x` would fail. The OpenCL package registers its own
# 8/16-wide classes the same way.
import pygf as _pygf

for _value in list(globals().values()):
    # `register_vector_type` reads `_fields_`, so only the concrete classes qualify; the
    # modules and the base classes in this namespace are filtered out here rather than
    # walked once more inside pygf.
    if isinstance(_value, type):
        _pygf.register_vector_type(_value)

del _value, _pygf