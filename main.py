from pygpgpu.opencl.runtime import cl
from ctypes import POINTER, byref, c_uint, c_void_p, c_char_p, c_size_t, c_int, c_float

platform = c_void_p()
num_platforms = c_uint()
ret = cl.clGetPlatformIDs(1, byref(platform), byref(num_platforms))
print(ret)