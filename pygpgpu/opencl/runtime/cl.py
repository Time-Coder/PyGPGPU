from __future__ import annotations
import ctypes
import os
import platform
from typing import Dict
from ctypes import c_int

from .clinfo import CLInfo
from .clconstantes import IntConstante


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
            error_code = IntConstante.get("CL_SUCCESS")
            if func_info["restype"] == c_int:
                try:
                    return_value = IntConstante.get(return_value)
                    error_code = return_value
                except:
                    pass
            
            if "errcode_ret" in func_info["args"]:
                if args_names is None:
                    args_names = list(func_info["args"].keys())

                idx = args_names.index("errcode_ret")
                error_code = IntConstante.get(args[idx].contents.value)

            if isinstance(error_code, IntConstante) and error_code.name in func_info["errors"]:
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
    pass