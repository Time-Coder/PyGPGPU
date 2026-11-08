# Buffer allocation flags, memory queries, and the command types.
from ....constants import IntEnum, IntFlag
from .core import cl_int, cl_ulong



class cl_mem_flags(IntFlag):
    CL_MEM_EXT_HOST_PTR_QCOM                    = (1 << 29)
    CL_MEM_USE_UNCACHED_CPU_MEMORY_IMG          = (1 << 26)
    CL_MEM_USE_CACHED_CPU_MEMORY_IMG            = (1 << 27)
    CL_MEM_USE_GRALLOC_PTR_IMG                  = (1 << 28)
    CL_MEM_NO_ACCESS_INTEL                      = (1 << 24)
    CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL      = (1 << 25)
    CL_MEM_FORCE_HOST_MEMORY_INTEL              = (1 << 20)
    CL_MEM_READ_WRITE                           = (1 << 0)
    CL_MEM_WRITE_ONLY                           = (1 << 1)
    CL_MEM_READ_ONLY                            = (1 << 2)
    CL_MEM_USE_HOST_PTR                         = (1 << 3)
    CL_MEM_ALLOC_HOST_PTR                       = (1 << 4)
    CL_MEM_COPY_HOST_PTR                        = (1 << 5)
    CL_MEM_HOST_WRITE_ONLY                      = (1 << 7)
    CL_MEM_HOST_READ_ONLY                       = (1 << 8)
    CL_MEM_HOST_NO_ACCESS                       = (1 << 9)
    CL_MEM_KERNEL_READ_AND_WRITE                = (1 << 12)

class cl_mem_properties(IntEnum):
    CL_MEM_ALLOC_FLAGS_IMG                             = 0x40D7
    CL_MEM_DEVICE_HANDLE_LIST_KHR                      = 0x2051
    CL_MEM_DEVICE_HANDLE_LIST_END_KHR                  = 0
    CL_MEM_LOCALLY_UNCACHED_RESOURCE_INTEL             = 0x4218
    CL_MEM_DEVICE_ID_INTEL                             = 0x4219

    @classmethod
    def dtype(cls):
        return cl_ulong
    
class cl_mem_info(IntEnum):
    CL_MEM_TYPE                                = 0x1100
    CL_MEM_FLAGS                               = 0x1101
    CL_MEM_SIZE                                = 0x1102
    CL_MEM_HOST_PTR                            = 0x1103
    CL_MEM_MAP_COUNT                           = 0x1104
    CL_MEM_REFERENCE_COUNT                     = 0x1105
    CL_MEM_CONTEXT                             = 0x1106
    CL_MEM_ASSOCIATED_MEMOBJECT                = 0x1107
    CL_MEM_OFFSET                              = 0x1108
    CL_MEM_USES_SVM_POINTER                    = 0x1109
    CL_MEM_PROPERTIES                          = 0x110A
    CL_MEM_D3D9_RESOURCE_NV                    = 0x4027
    CL_MEM_D3D10_RESOURCE_NV                   = 0x4015
    CL_MEM_D3D10_RESOURCE_KHR                  = 0x4015
    CL_MEM_D3D11_RESOURCE_NV                   = 0x401E
    CL_MEM_D3D11_RESOURCE_KHR                  = 0x401E
    CL_MEM_DX9_MEDIA_ADAPTER_TYPE_KHR          = 0x2028
    CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR          = 0x2029
    CL_MEM_DX9_RESOURCE_INTEL                  = 0x4027
    CL_MEM_DX9_SHARED_HANDLE_INTEL             = 0x4074
    CL_MEM_USES_SVM_POINTER_ARM                = 0x40B7

class cl_mem_object_type(IntEnum):
    CL_MEM_OBJECT_BUFFER                       = 0x10F0
    CL_MEM_OBJECT_IMAGE2D                      = 0x10F1
    CL_MEM_OBJECT_IMAGE3D                      = 0x10F2
    CL_MEM_OBJECT_IMAGE2D_ARRAY                = 0x10F3
    CL_MEM_OBJECT_IMAGE1D                      = 0x10F4
    CL_MEM_OBJECT_IMAGE1D_ARRAY                = 0x10F5
    CL_MEM_OBJECT_IMAGE1D_BUFFER               = 0x10F6
    CL_MEM_OBJECT_PIPE                         = 0x10F7

class cl_event_info(IntEnum):
    CL_EVENT_COMMAND_QUEUE                     = 0x11D0
    CL_EVENT_COMMAND_TYPE                      = 0x11D1
    CL_EVENT_REFERENCE_COUNT                   = 0x11D2
    CL_EVENT_COMMAND_EXECUTION_STATUS          = 0x11D3
    CL_EVENT_CONTEXT                           = 0x11D4
    CL_EVENT_COMMAND_TERMINATION_REASON_ARM    = 0x41ED

class cl_command_execution_status(IntEnum):
    CL_COMPLETE                                = 0x0
    CL_RUNNING                                 = 0x1
    CL_SUBMITTED                               = 0x2
    CL_QUEUED                                  = 0x3

    @classmethod
    def dtype(cls):
        return cl_int
    
class cl_command_type(IntEnum):
    CL_COMMAND_NDRANGE_KERNEL                  = 0x11F0
    CL_COMMAND_TASK                            = 0x11F1
    CL_COMMAND_NATIVE_KERNEL                   = 0x11F2
    CL_COMMAND_READ_BUFFER                     = 0x11F3
    CL_COMMAND_WRITE_BUFFER                    = 0x11F4
    CL_COMMAND_COPY_BUFFER                     = 0x11F5
    CL_COMMAND_READ_IMAGE                      = 0x11F6
    CL_COMMAND_WRITE_IMAGE                     = 0x11F7
    CL_COMMAND_COPY_IMAGE                      = 0x11F8
    CL_COMMAND_COPY_IMAGE_TO_BUFFER            = 0x11F9
    CL_COMMAND_COPY_BUFFER_TO_IMAGE            = 0x11FA
    CL_COMMAND_MAP_BUFFER                      = 0x11FB
    CL_COMMAND_MAP_IMAGE                       = 0x11FC
    CL_COMMAND_UNMAP_MEM_OBJECT                = 0x11FD
    CL_COMMAND_MARKER                          = 0x11FE
    CL_COMMAND_ACQUIRE_GL_OBJECTS              = 0x11FF
    CL_COMMAND_RELEASE_GL_OBJECTS              = 0x1200
    CL_COMMAND_READ_BUFFER_RECT                = 0x1201
    CL_COMMAND_WRITE_BUFFER_RECT               = 0x1202
    CL_COMMAND_COPY_BUFFER_RECT                = 0x1203
    CL_COMMAND_USER                            = 0x1204
    CL_COMMAND_BARRIER                         = 0x1205
    CL_COMMAND_MIGRATE_MEM_OBJECTS             = 0x1206
    CL_COMMAND_FILL_BUFFER                     = 0x1207
    CL_COMMAND_FILL_IMAGE                      = 0x1208
    CL_COMMAND_SVM_FREE                        = 0x1209
    CL_COMMAND_SVM_MEMCPY                      = 0x120A
    CL_COMMAND_SVM_MEMFILL                     = 0x120B
    CL_COMMAND_SVM_MAP                         = 0x120C
    CL_COMMAND_SVM_UNMAP                       = 0x120D
    CL_COMMAND_SVM_MIGRATE_MEM                 = 0x120E
    CL_COMMAND_GL_FENCE_SYNC_OBJECT_KHR                = 0x200D
    CL_COMMAND_COMMAND_BUFFER_KHR                      = 0x12A8
    CL_COMMAND_MIGRATE_MEM_OBJECT_EXT                  = 0x4040
    CL_COMMAND_ACQUIRE_GRALLOC_OBJECTS_IMG             = 0x40D2
    CL_COMMAND_RELEASE_GRALLOC_OBJECTS_IMG             = 0x40D3
    CL_COMMAND_GENERATE_MIPMAP_IMG                     = 0x40D6
    CL_COMMAND_ACQUIRE_EXTERNAL_MEM_OBJECTS_KHR        = 0x2047
    CL_COMMAND_RELEASE_EXTERNAL_MEM_OBJECTS_KHR        = 0x2048
    CL_COMMAND_SEMAPHORE_WAIT_KHR                      = 0x2042
    CL_COMMAND_SEMAPHORE_SIGNAL_KHR                    = 0x2043
    CL_COMMAND_SVM_FREE_ARM                            = 0x40BA
    CL_COMMAND_SVM_MEMCPY_ARM                          = 0x40BB
    CL_COMMAND_SVM_MEMFILL_ARM                         = 0x40BC
    CL_COMMAND_SVM_MAP_ARM                             = 0x40BD
    CL_COMMAND_SVM_UNMAP_ARM                           = 0x40BE
    CL_COMMAND_MEMFILL_INTEL                           = 0x4204
    CL_COMMAND_MEMCPY_INTEL                            = 0x4205
    CL_COMMAND_MIGRATEMEM_INTEL                        = 0x4206
    CL_COMMAND_MEMADVISE_INTEL                         = 0x4207
    CL_COMMAND_ACQUIRE_DX9_MEDIA_SURFACES_KHR          = 0x202B
    CL_COMMAND_RELEASE_DX9_MEDIA_SURFACES_KHR          = 0x202C
    CL_COMMAND_ACQUIRE_DX9_OBJECTS_INTEL               = 0x402A
    CL_COMMAND_RELEASE_DX9_OBJECTS_INTEL               = 0x402B
    CL_COMMAND_ACQUIRE_D3D11_OBJECTS_KHR               = 0x4020
    CL_COMMAND_RELEASE_D3D11_OBJECTS_KHR               = 0x4021
    CL_COMMAND_ACQUIRE_D3D11_OBJECTS_NV                = 0x4020
    CL_COMMAND_RELEASE_D3D11_OBJECTS_NV                = 0x4021
    CL_COMMAND_ACQUIRE_D3D10_OBJECTS_KHR               = 0x4017
    CL_COMMAND_RELEASE_D3D10_OBJECTS_KHR               = 0x4018
    CL_COMMAND_ACQUIRE_D3D10_OBJECTS_NV                = 0x4017
    CL_COMMAND_RELEASE_D3D10_OBJECTS_NV                = 0x4018
    CL_COMMAND_ACQUIRE_D3D9_OBJECTS_NV                 = 0x402A
    CL_COMMAND_RELEASE_D3D9_OBJECTS_NV                 = 0x402B
