from pygpgpu.opencl.runtime import CL, cl_device_type
from ctypes import POINTER, byref, c_uint, c_void_p, c_char_p, c_size_t, c_int, c_float

platform = c_void_p()
num_platforms = c_uint()
ret = CL.clGetPlatformIDs(1, byref(platform), byref(num_platforms))

device = c_void_p()
num_devices = c_uint()
ret = CL.clGetDeviceIDs(platform, cl_device_type.CL_DEVICE_TYPE_ALL, 1, byref(device), byref(num_devices))

context = CL.clCreateContext(None, 1, byref(device), None, None, byref(c_int()))
