# Image formats, channel orders, and sampler queries.
from ....constants import IntEnum
from .core import cl_ulong



class cl_channel_order(IntEnum):
    CL_R                                       = 0x10B0
    CL_A                                       = 0x10B1
    CL_RG                                      = 0x10B2
    CL_RA                                      = 0x10B3
    CL_RGB                                     = 0x10B4
    CL_RGBA                                    = 0x10B5
    CL_BGRA                                    = 0x10B6
    CL_ARGB                                    = 0x10B7
    CL_INTENSITY                               = 0x10B8
    CL_LUMINANCE                               = 0x10B9
    CL_Rx                                      = 0x10BA
    CL_RGx                                     = 0x10BB
    CL_RGBx                                    = 0x10BC
    CL_DEPTH                                   = 0x10BD
    CL_sRGB                                    = 0x10BF
    CL_sRGBx                                   = 0x10C0
    CL_sRGBA                                   = 0x10C1
    CL_sBGRA                                   = 0x10C2
    CL_ABGR                                    = 0x10C3
    CL_DEPTH_STENCIL                           = 0x10BE
    CL_NV21_IMG                                = 0x40D0
    CL_YV12_IMG                                = 0x40D1
    CL_YUYV_INTEL                              = 0x4076
    CL_UYVY_INTEL                              = 0x4077
    CL_YVYU_INTEL                              = 0x4078
    CL_VYUY_INTEL                              = 0x4079
    CL_NV12_INTEL                              = 0x410E

class cl_channel_type(IntEnum):
    CL_SNORM_INT8                              = 0x10D0
    CL_SNORM_INT16                             = 0x10D1
    CL_UNORM_INT8                              = 0x10D2
    CL_UNORM_INT16                             = 0x10D3
    CL_UNORM_SHORT_565                         = 0x10D4
    CL_UNORM_SHORT_555                         = 0x10D5
    CL_UNORM_INT_101010                        = 0x10D6
    CL_SIGNED_INT8                             = 0x10D7
    CL_SIGNED_INT16                            = 0x10D8
    CL_SIGNED_INT32                            = 0x10D9
    CL_UNSIGNED_INT8                           = 0x10DA
    CL_UNSIGNED_INT16                          = 0x10DB
    CL_UNSIGNED_INT32                          = 0x10DC
    CL_HALF_FLOAT                              = 0x10DD
    CL_FLOAT                                   = 0x10DE
    CL_UNORM_INT_101010_2                      = 0x10E0
    CL_UNORM_INT24                             = 0x10DF
    CL_UNSIGNED_INT_RAW10_EXT                  = 0x10E3
    CL_UNSIGNED_INT_RAW12_EXT                  = 0x10E4

class cl_addressing_mode(IntEnum):
    CL_ADDRESS_NONE                           = 0x1130
    CL_ADDRESS_CLAMP_TO_EDGE                  = 0x1131
    CL_ADDRESS_CLAMP                          = 0x1132
    CL_ADDRESS_REPEAT                         = 0x1133
    CL_ADDRESS_MIRRORED_REPEAT                = 0x1134

class cl_filter_mode(IntEnum):
    CL_FILTER_NEAREST                         = 0x1140
    CL_FILTER_LINEAR                          = 0x1141

class cl_sampler_info(IntEnum):
    CL_SAMPLER_REFERENCE_COUNT                 = 0x1150
    CL_SAMPLER_CONTEXT                         = 0x1151
    CL_SAMPLER_NORMALIZED_COORDS               = 0x1152
    CL_SAMPLER_ADDRESSING_MODE                 = 0x1153
    CL_SAMPLER_FILTER_MODE                     = 0x1154
    CL_SAMPLER_MIP_FILTER_MODE                 = 0x1155
    CL_SAMPLER_LOD_MIN                         = 0x1156
    CL_SAMPLER_LOD_MAX                         = 0x1157
    CL_SAMPLER_PROPERTIES                      = 0x1158

class cl_sampler_properties(IntEnum):
    CL_SAMPLER_MIP_FILTER_MODE_KHR                     = 0x1155
    CL_SAMPLER_LOD_MIN_KHR                             = 0x1156
    CL_SAMPLER_LOD_MAX_KHR                             = 0x1157

    @classmethod
    def dtype(cls)->type:
        return cl_ulong
