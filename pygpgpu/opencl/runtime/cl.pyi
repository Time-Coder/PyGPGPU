from ctypes import c_void_p, c_size_t, c_int64, c_char_p
from typing import Any

from .cltypes import (
    ErrorCode,
    cl_uint,
    cl_platform_id,
    cl_program,
    cl_device_id,
    ptr_ptr_char,
    cl_device_info,
    ptr_cl_platform_id,
    ptr_cl_uint,
    ptr_size_t,
    cl_platform_info,
    cl_device_type,
    ptr_cl_device_id,
    ptr_cl_int,
    ptr_int64,
    CL_CONTEXT_NOTIFY_CALLBACK,
    CL_BULD_PROGRAM_CALLBACK,
    cl_context,
    cl_context_info,
    cl_program_info,
    cl_program_build_info
)


class CL:

    class Func:

        def __call__(self, *args, **kwargs)->Any: ...

    @property
    def check_error(self)->bool: ...
    
    @check_error.setter
    def check_error(self, flag:bool)->None: ...

    @property
    def print_call(self)->bool: ...
    
    @print_call.setter
    def print_call(self, flag:bool)->None: ...

    def clGetPlatformIDs(num_entries: cl_uint, platforms: ptr_cl_platform_id, num_platforms: ptr_cl_uint)->ErrorCode: ...
    def clGetPlatformInfo(platform: cl_platform_id, param_name: cl_platform_info, param_value_size: c_size_t, param_value: c_void_p, param_value_size_ret: ptr_size_t)->ErrorCode: ...
    def clGetDeviceIDs(platform: cl_platform_id, device_type: cl_device_type, num_entries: cl_uint, devices: ptr_cl_device_id, num_devices: ptr_cl_uint)->ErrorCode: ...
    def clGetDeviceInfo(device:cl_device_id, param_name:cl_device_info, param_value_size:c_size_t, param_value:c_void_p, param_value_size_ret:ptr_size_t)->ErrorCode: ...
    def clCreateContext(properties:ptr_int64, num_devices:cl_uint, devices:ptr_cl_device_id, pfn_notify:CL_CONTEXT_NOTIFY_CALLBACK, user_data:c_void_p, errcode_ret:ptr_cl_int)->cl_context: ...
    def clReleaseContext(context:cl_context)->ErrorCode: ...
    def clGetContextInfo(context:cl_context, param_name:cl_context_info, param_value_size:c_size_t, param_value:c_void_p, param_value_size_ret:ptr_size_t)->ErrorCode:...
    def clCreateProgramWithSource(context:cl_context, count:cl_uint, strings:ptr_ptr_char, lengths:ptr_size_t, errcode_ret:ptr_cl_int)->cl_program:...
    def clBuildProgram(program:cl_program, num_devices:cl_uint, device_list:ptr_cl_device_id, options:c_char_p, pfn_notify:CL_BULD_PROGRAM_CALLBACK, user_data:c_void_p)->ErrorCode:...
    def clGetProgramInfo(program:cl_program, param_name:cl_program_info, param_value_size:c_size_t, param_value:c_void_p, param_value_size_ret:ptr_size_t)->ErrorCode:...
    def clGetProgramBuildInfo(program:cl_program, device:cl_device_id, param_name:cl_program_build_info, param_value_size:c_size_t, param_value:c_void_p, param_value_size_ret:ptr_size_t)->ErrorCode:...
    def clReleaseProgram(program:cl_program)->ErrorCode:...
