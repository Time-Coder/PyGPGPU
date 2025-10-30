from __future__ import annotations
import ctypes
import os
import platform
from typing import Dict

from .CLConstante import CLConstante, define_constantes
from .CLInfo import CLInfo


@define_constantes
class CL:

    class Func:

        def __init__(self, parent:CL, name:str):
            self.parent:CL = parent
            self.name = name

        def __call__(self, *args, **kwargs):
            func_info = CLInfo.func_signatures[self.name]
            func = func_info["dll_func"]
            if func is None:
                func = getattr(self.parent.dll, self.name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func
            return func(*args, **kwargs)

    def __init__(self):
        self.__dll = None
        self.__func_map:Dict[str, CL.Func] = {}

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
    
    @property
    def dll(self):
        if self.__dll is None:
            self.__dll = ctypes.CDLL(self.opencl_lib_path())

        return self.__dll

    def __getattr__(self, name:str)->Func:
        if name not in self.__func_map:
            self.__func_map[name] = CL.Func(self, name)

        return self.__func_map[name]

cl:CL = CL()