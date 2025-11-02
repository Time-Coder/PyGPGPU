from __future__ import annotations
import ctypes
import os
import platform
from typing import Dict, Callable
from ctypes import c_int

from .clinfo import CLInfo
from .cltypes import ErrorCode


class MetaCL(type):

    __dll = None
    __func_map:Dict[str, Callable] = {}

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

    def __getattr__(self, name:str)->Callable:
        if name not in MetaCL.__func_map:
            MetaCL.__func_map[name] = self.Func(name)

        return MetaCL.__func_map[name]


class CL(metaclass=MetaCL):

    print_call:bool = False
    check_error:bool = True

    class Func:

        def __init__(self, name:str):
            self.name = name

        def __call__(self, *args, **kwargs):
            if CL.print_call:
                call_str = self.name + "(" + ", ".join([self.__str_arg(arg) for arg in args] + [f"{key}={self.__str_arg(value)}" for key, value in kwargs.items()]) + ")"
                print(call_str, end="", flush=True)

            func_info = CLInfo.func_signatures[self.name]
            if "dll_func" not in func_info:
                func = getattr(MetaCL.dll(), self.name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func
            else:
                func = func_info["dll_func"]
            
            current_n_args:int = len(args)
            target_n_args:int = len(func_info["args"])
            args_names = None
            if current_n_args < target_n_args:
                args = list(args)
                args_names = list(func_info["args"].keys())
                for i in range(current_n_args, target_n_args):
                    args.append(kwargs[args_names[i]])

            return_value = func(*args)
            error_code = ErrorCode.CL_SUCCESS
            if func_info["restype"] == c_int:
                try:
                    return_value = ErrorCode(return_value)
                    error_code = return_value
                except:
                    pass
            
            if CL.print_call:
                print(f"->{return_value}", flush=True)

            if CL.check_error:
                if "errcode_ret" in func_info["args"]:
                    if args_names is None:
                        args_names = list(func_info["args"].keys())

                    idx = args_names.index("errcode_ret")
                    error_code = ErrorCode(args[idx].contents.value)

                if error_code != ErrorCode.CL_SUCCESS:
                    if error_code.name in func_info["errors"]:
                        raise RuntimeError(f"{error_code}: {func_info['errors'][error_code.name]}")
                    else:
                        raise RuntimeError(f"{error_code}: unknown error.")
            
            return return_value

        @staticmethod
        def __get_ctypes_type_name(tp) -> str:
            if isinstance(tp, type):
                name = tp.__name__
            else:
                name = type(tp).__name__

            name_map = {
                'c_char': 'char',
                'c_wchar': 'wchar',
                'c_byte': 'char',
                'c_ubyte': 'uchar',
                'c_short': 'short',
                'c_ushort': 'ushort',
                'c_int': 'int',
                'c_uint': 'uint',
                'c_long': 'long',
                'c_ulong': 'ulong',
                'c_longlong': 'int64_t',
                'c_ulonglong': 'uint64_t',
                'c_float': 'float',
                'c_double': 'double',
                'c_bool': 'bool',
                'c_void_p': 'void*'
            }
            return name_map[name]

        @staticmethod
        def __str_arg(arg)->str:
            if type(arg).__name__ == 'CArgObject':
                obj = arg._obj
                type_name = CL.Func.__get_ctypes_type_name(obj)
                addr = ctypes.addressof(obj)
                return f"({type_name}*)(0x{addr:x})"

            if isinstance(arg, ctypes._Pointer):
                pointed_type = arg._type_
                type_name = CL.Func.__get_ctypes_type_name(pointed_type)
                addr = ctypes.addressof(arg.contents) if arg else 0
                return f"({type_name}*)(0x{addr:x})"

            if isinstance(arg, ctypes.Array):
                elem_type = arg._type_
                elem_type_name = CL.Func.__get_ctypes_type_name(elem_type)
                length = len(arg)
                addr = ctypes.addressof(arg)
                return f"({elem_type_name}[{length}])(0x{addr:x})"

            if hasattr(arg, 'value'):
                return str(arg.value)
            
            if arg is None:
                return "NULL"

            return str(arg)