from ctypes import c_uint, c_int, c_void_p, POINTER
from .CLConstante import CLConstante


class CL:

    def clGetPlatformIDs(num_entries:c_uint, platforms:POINTER, num_platforms:POINTER)->CLConstante: ...