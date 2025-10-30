from ctypes import c_uint, c_int, c_ulong, c_void_p, POINTER
from .CLConstante import CLConstante


class CL:

    def clGetPlatformIDs(num_entries:c_uint, platforms:POINTER, num_platforms:POINTER)->CLConstante: ...
    def clGetDeviceIDs(platform:c_void_p, device_type:c_ulong, num_entries:c_uint, devices:POINTER, num_devices:POINTER)->CLConstante: ...