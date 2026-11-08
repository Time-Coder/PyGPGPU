# The scalar aliases, the opaque handles and the callback pointer types.
from ctypes import c_uint64, c_ubyte, c_uint32, c_int32, c_int64, c_void_p, c_size_t, c_char_p, POINTER, CFUNCTYPE
from typing_extensions import TypeAlias



cl_int = c_int32
cl_uint = c_uint32
cl_ulong = c_uint64
cl_long = c_int64

cl_platform_id = c_void_p
cl_device_id = c_void_p
cl_context = c_void_p
cl_program = c_void_p
cl_kernel = c_void_p
cl_command_queue = c_void_p
cl_mem = c_void_p
cl_sampler = c_void_p

cl_bitfield = cl_ulong
cl_semaphore_type_khr = cl_uint
cl_properties = cl_ulong
cl_event = c_void_p

ptr_cl_platform_id:TypeAlias = POINTER(cl_platform_id)
ptr_cl_uint:TypeAlias = POINTER(cl_uint)
ptr_cl_int:TypeAlias = POINTER(cl_int)
ptr_size_t:TypeAlias = POINTER(c_size_t)
ptr_cl_device_id:TypeAlias = POINTER(cl_device_id)
ptr_int64:TypeAlias = POINTER(c_int64)
ptr_ptr_char:TypeAlias = POINTER(c_char_p)
ptr_ubyte:TypeAlias = POINTER(c_ubyte)
ptr_ptr_ubyte:TypeAlias = POINTER(ptr_ubyte)
ptr_cl_kernel:TypeAlias = POINTER(cl_kernel)
ptr_cl_ulong:TypeAlias = POINTER(cl_ulong)
ptr_cl_event:TypeAlias = POINTER(cl_event)
ptr_cl_mem:TypeAlias = POINTER(cl_mem)
ptr_ptr_void:TypeAlias = POINTER(c_void_p)
CL_CONTEXT_NOTIFY_CALLBACK:TypeAlias = CFUNCTYPE(None, c_char_p, c_void_p, c_size_t, c_void_p)
CL_BULD_PROGRAM_CALLBACK:TypeAlias = CFUNCTYPE(None, cl_program, c_void_p)
CL_EVENT_NOTIFY_CALLBACK:TypeAlias = CFUNCTYPE(None, cl_event, cl_int, c_void_p)
