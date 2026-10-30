from .char2 import char2
from .char4 import char4
from .char8 import char8
from .char16 import char16

from .uchar2 import uchar2
from .uchar4 import uchar4
from .uchar8 import uchar8
from .uchar16 import uchar16

from .short2 import short2
from .short4 import short4
from .short8 import short8
from .short16 import short16

from .ushort2 import ushort2
from .ushort4 import ushort4
from .ushort8 import ushort8
from .ushort16 import ushort16

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

from .float8 import float8
from .float16 import float16

from .double8 import double8
from .double16 import double16

from .int8 import int8
from .int16 import int16

from .uint8 import uint8
from .uint16 import uint16

from .genVec import genVec
from .genVec2 import genVec2
from .genVec3 import genVec3
from .genVec4 import genVec4
from .genVec8 import genVec8
from .genVec16 import genVec16

# pygf owns the numpy patches and the `VectorArray` view they produce (see
# pygf/vector_array.py). It knows the types it defines; the widths above that PyGPGPU adds --
# the 8/16-wide ones and the char/uchar/short families -- have to be handed over, or an array
# of them would come back as plain records and `a[i].x` would fail.
import pygf as _pygf

for _value in list(globals().values()):
    _pygf.register_vector_type(_value)

del _value, _pygf