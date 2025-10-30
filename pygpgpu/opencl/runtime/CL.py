from __future__ import annotations
import ctypes
import os
import platform
from typing import Dict
from ctypes import c_int

from .CLConstante import CLConstante
from .CLInfo import CLInfo


class MetaCL(type):

    class Func:

        def __init__(self, name:str):
            self.name = name

        def __call__(self, *args, **kwargs):
            func_info = CLInfo.func_signatures[self.name]
            func = func_info["dll_func"]
            if func is None:
                func = getattr(MetaCL.dll(), self.name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func
            
            current_n_args:int = len(args)
            target_n_args:int = len(func_info["args"])
            args_names = None
            if current_n_args < target_n_args:
                args = list(args)
                args_names = list(func_info["args"].keys())
                for i in range(current_n_args, target_n_args):
                    args.append(kwargs[args_names[i]])

            return_value = func(*args)
            error_code = CLConstante.get("CL_SUCCESS")
            if func_info["restype"] == c_int:
                try:
                    return_value = CLConstante.get(return_value)
                    error_code = return_value
                except:
                    pass
            
            if "errcode_ret" in func_info["args"]:
                if args_names is None:
                    args_names = list(func_info["args"].keys())

                idx = args_names.index("errcode_ret")
                error_code = CLConstante.get(args[idx].contents.value)

            if isinstance(error_code, CLConstante) and error_code.name in func_info["errors"]:
                raise RuntimeError(f"{error_code}: {func_info['errors'][error_code.name]}")
                
            return return_value

    __dll = None
    __func_map:Dict[str, Func] = {}

    @staticmethod
    def opencl_lib_path():
        system = platform.system()
        
        if system == "Windows":
            return "OpenCL.dll"
        
        elif system == "Darwin":
            return "/System/Library/Frameworks/OpenCL.framework/OpenCL"
        
        elif system == "Linux":
            possible_paths = [
                "libOpenCL.so",
                "/usr/lib/x86_64-linux-gnu/libOpenCL.so",
                "/usr/lib64/libOpenCL.so",
                "/usr/lib/libOpenCL.so",
                "/opt/ocl/lib/libOpenCL.so",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
                
            return "libOpenCL.so"
    
    @staticmethod
    def dll():
        if MetaCL.__dll is None:
            MetaCL.__dll = ctypes.CDLL(MetaCL.opencl_lib_path())

        return MetaCL.__dll

    def __getattr__(self, name:str)->Func:
        if name not in MetaCL.__func_map:
            MetaCL.__func_map[name] = MetaCL.Func(name)

        return MetaCL.__func_map[name]


class CL(metaclass=MetaCL):
    # Error Codes
    CL_SUCCESS = CLConstante("CL_SUCCESS", 0)
    CL_DEVICE_NOT_FOUND = CLConstante("CL_DEVICE_NOT_FOUND", -1)
    CL_DEVICE_NOT_AVAILABLE = CLConstante("CL_DEVICE_NOT_AVAILABLE", -2)
    CL_COMPILER_NOT_AVAILABLE = CLConstante("CL_COMPILER_NOT_AVAILABLE", -3)
    CL_MEM_OBJECT_ALLOCATION_FAILURE = CLConstante("CL_MEM_OBJECT_ALLOCATION_FAILURE", -4)
    CL_OUT_OF_RESOURCES = CLConstante("CL_OUT_OF_RESOURCES", -5)
    CL_OUT_OF_HOST_MEMORY = CLConstante("CL_OUT_OF_HOST_MEMORY", -6)
    CL_PROFILING_INFO_NOT_AVAILABLE = CLConstante("CL_PROFILING_INFO_NOT_AVAILABLE", -7)
    CL_MEM_COPY_OVERLAP = CLConstante("CL_MEM_COPY_OVERLAP", -8)
    CL_IMAGE_FORMAT_MISMATCH = CLConstante("CL_IMAGE_FORMAT_MISMATCH", -9)
    CL_IMAGE_FORMAT_NOT_SUPPORTED = CLConstante("CL_IMAGE_FORMAT_NOT_SUPPORTED", -10)
    CL_BUILD_PROGRAM_FAILURE = CLConstante("CL_BUILD_PROGRAM_FAILURE", -11)
    CL_MAP_FAILURE = CLConstante("CL_MAP_FAILURE", -12)
    CL_MISALIGNED_SUB_BUFFER_OFFSET = CLConstante("CL_MISALIGNED_SUB_BUFFER_OFFSET", -13)
    CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = CLConstante("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST", -14)
    CL_COMPILE_PROGRAM_FAILURE = CLConstante("CL_COMPILE_PROGRAM_FAILURE", -15)
    CL_LINKER_NOT_AVAILABLE = CLConstante("CL_LINKER_NOT_AVAILABLE", -16)
    CL_LINK_PROGRAM_FAILURE = CLConstante("CL_LINK_PROGRAM_FAILURE", -17)
    CL_DEVICE_PARTITION_FAILED = CLConstante("CL_DEVICE_PARTITION_FAILED", -18)
    CL_KERNEL_ARG_INFO_NOT_AVAILABLE = CLConstante("CL_KERNEL_ARG_INFO_NOT_AVAILABLE", -19)
    CL_INVALID_VALUE = CLConstante("CL_INVALID_VALUE", -30)
    CL_INVALID_DEVICE_TYPE = CLConstante("CL_INVALID_DEVICE_TYPE", -31)
    CL_INVALID_PLATFORM = CLConstante("CL_INVALID_PLATFORM", -32)
    CL_INVALID_DEVICE = CLConstante("CL_INVALID_DEVICE", -33)
    CL_INVALID_CONTEXT = CLConstante("CL_INVALID_CONTEXT", -34)
    CL_INVALID_QUEUE_PROPERTIES = CLConstante("CL_INVALID_QUEUE_PROPERTIES", -35)
    CL_INVALID_COMMAND_QUEUE = CLConstante("CL_INVALID_COMMAND_QUEUE", -36)
    CL_INVALID_HOST_PTR = CLConstante("CL_INVALID_HOST_PTR", -37)
    CL_INVALID_MEM_OBJECT = CLConstante("CL_INVALID_MEM_OBJECT", -38)
    CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = CLConstante("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", -39)
    CL_INVALID_IMAGE_SIZE = CLConstante("CL_INVALID_IMAGE_SIZE", -40)
    CL_INVALID_SAMPLER = CLConstante("CL_INVALID_SAMPLER", -41)
    CL_INVALID_BINARY = CLConstante("CL_INVALID_BINARY", -42)
    CL_INVALID_BUILD_OPTIONS = CLConstante("CL_INVALID_BUILD_OPTIONS", -43)
    CL_INVALID_PROGRAM = CLConstante("CL_INVALID_PROGRAM", -44)
    CL_INVALID_PROGRAM_EXECUTABLE = CLConstante("CL_INVALID_PROGRAM_EXECUTABLE", -45)
    CL_INVALID_KERNEL_NAME = CLConstante("CL_INVALID_KERNEL_NAME", -46)
    CL_INVALID_KERNEL_DEFINITION = CLConstante("CL_INVALID_KERNEL_DEFINITION", -47)
    CL_INVALID_KERNEL = CLConstante("CL_INVALID_KERNEL", -48)
    CL_INVALID_ARG_INDEX = CLConstante("CL_INVALID_ARG_INDEX", -49)
    CL_INVALID_ARG_VALUE = CLConstante("CL_INVALID_ARG_VALUE", -50)
    CL_INVALID_ARG_SIZE = CLConstante("CL_INVALID_ARG_SIZE", -51)
    CL_INVALID_KERNEL_ARGS = CLConstante("CL_INVALID_KERNEL_ARGS", -52)
    CL_INVALID_WORK_DIMENSION = CLConstante("CL_INVALID_WORK_DIMENSION", -53)
    CL_INVALID_WORK_GROUP_SIZE = CLConstante("CL_INVALID_WORK_GROUP_SIZE", -54)
    CL_INVALID_WORK_ITEM_SIZE = CLConstante("CL_INVALID_WORK_ITEM_SIZE", -55)
    CL_INVALID_GLOBAL_OFFSET = CLConstante("CL_INVALID_GLOBAL_OFFSET", -56)
    CL_INVALID_EVENT_WAIT_LIST = CLConstante("CL_INVALID_EVENT_WAIT_LIST", -57)
    CL_INVALID_EVENT = CLConstante("CL_INVALID_EVENT", -58)
    CL_INVALID_OPERATION = CLConstante("CL_INVALID_OPERATION", -59)
    CL_INVALID_GL_OBJECT = CLConstante("CL_INVALID_GL_OBJECT", -60)
    CL_INVALID_BUFFER_SIZE = CLConstante("CL_INVALID_BUFFER_SIZE", -61)
    CL_INVALID_MIP_LEVEL = CLConstante("CL_INVALID_MIP_LEVEL", -62)
    CL_INVALID_GLOBAL_WORK_SIZE = CLConstante("CL_INVALID_GLOBAL_WORK_SIZE", -63)
    CL_INVALID_PROPERTY = CLConstante("CL_INVALID_PROPERTY", -64)
    CL_INVALID_IMAGE_DESCRIPTOR = CLConstante("CL_INVALID_IMAGE_DESCRIPTOR", -65)
    CL_INVALID_COMPILER_OPTIONS = CLConstante("CL_INVALID_COMPILER_OPTIONS", -66)
    CL_INVALID_LINKER_OPTIONS = CLConstante("CL_INVALID_LINKER_OPTIONS", -67)
    CL_INVALID_DEVICE_PARTITION_COUNT = CLConstante("CL_INVALID_DEVICE_PARTITION_COUNT", -68)
    CL_INVALID_PIPE_SIZE = CLConstante("CL_INVALID_PIPE_SIZE", -69)
    CL_INVALID_DEVICE_QUEUE = CLConstante("CL_INVALID_DEVICE_QUEUE", -70)
    CL_INVALID_SPEC_ID = CLConstante("CL_INVALID_SPEC_ID", -71)
    CL_MAX_SIZE_RESTRICTION_EXCEEDED = CLConstante("CL_MAX_SIZE_RESTRICTION_EXCEEDED", -72)

    # cl_device_type
    CL_DEVICE_TYPE_DEFAULT = CLConstante("CL_DEVICE_TYPE_DEFAULT", (1 << 0))
    CL_DEVICE_TYPE_CPU = CLConstante("CL_DEVICE_TYPE_CPU", (1 << 1))
    CL_DEVICE_TYPE_GPU = CLConstante("CL_DEVICE_TYPE_GPU", (1 << 2))
    CL_DEVICE_TYPE_ACCELERATOR = CLConstante("CL_DEVICE_TYPE_ACCELERATOR", (1 << 3))
    CL_DEVICE_TYPE_CUSTOM = CLConstante("CL_DEVICE_TYPE_CUSTOM", (1 << 4))
    CL_DEVICE_TYPE_ALL = CLConstante("CL_DEVICE_TYPE_ALL", 0xFFFFFFFF)