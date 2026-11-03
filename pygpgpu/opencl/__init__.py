
from .driver import (
    image1d_t,
    image2d_t,
    image3d_t,
    image1d_array_t,
    image2d_array_t,
    sampler_t,
    char2, char3, char4, char8, char16,
    uchar2, uchar3, uchar4, uchar8, uchar16,
    short2, short3, short4, short8, short16,
    ushort2, ushort3, ushort4, ushort8, ushort16,
    int2, int3, int4, int8, int16,
    uint2, uint3, uint4, uint8, uint16,
    long2, long3, long4, long8, long16,
    ulong2, ulong3, ulong4, ulong8, ulong16,
    float2, float3, float4, float8, float16,
    double2, double3, double4, double8, double16,
    CL
)
from .driver import _start_cl_init

# The runtime is initialized on a background thread: `import pygpgpu.opencl` returns without
# waiting for the DLL load and the platform-count preheat. Every driver call waits for the
# initialization it needs (see `Func.__call__` in the driver), so nothing observes a
# half-initialized runtime. A runtime the machine lacks is recorded by the driver's init,
# not raised here.
_start_cl_init()