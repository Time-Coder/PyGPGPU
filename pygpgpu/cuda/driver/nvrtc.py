from __future__ import annotations
import platform
import shutil
import glob
import os
from typing import Dict, Callable, List
from ctypes import c_int, c_void_p, pointer, CDLL, addressof, _Pointer, Array

from .nvrtcinfo import NVRTCInfo
from .nvrtctypes import nvrtcResult


class MetaNVRTC(type):

    __dll = None
    __func_map:Dict[str, Callable] = {}

    @staticmethod
    def nvrtc_lib_path():
        system = platform.system()
        if system == "Windows":
            nvcc_path:str = shutil.which("nvcc.exe")
            if not nvcc_path:
                raise FileNotFoundError("nvcc.exe")
            
            nvcc_folder:str = os.path.dirname(nvcc_path).replace("\\", "/")
            nvrtc_candidates:List[str] = glob.glob(nvcc_folder + "/nvrtc64_*.dll")
            if not nvrtc_candidates:
                raise FileNotFoundError("nvrtc64_*.dll")
            
            lib_name = max(nvrtc_candidates)
        elif system == "Linux":
            lib_name = "libnvrtc.so"
        elif system == "Darwin":  # macOS
            lib_name = "libnvrtc.dylib"
        else:
            raise RuntimeError(f"unsupported operating system: {system}")
        
        return lib_name
    
    @staticmethod
    def dll():
        if MetaNVRTC.__dll is None:
            MetaNVRTC.__dll = CDLL(MetaNVRTC.nvrtc_lib_path())

        return MetaNVRTC.__dll

    def __getattr__(self, name:str)->Callable:
        if name not in MetaNVRTC.__func_map:
            MetaNVRTC.__func_map[name] = self.Func(name)

        return MetaNVRTC.__func_map[name]


class NVRTC(metaclass=MetaNVRTC):

    print_call:bool = False
    print_info:bool = True
    check_error:bool = True

    class Func:

        def __init__(self, name:str):
            self.name = name

        def __call__(self, *args, **kwargs):
            if NVRTC.print_call:
                call_str = self.name + "(" + ", ".join([self.__str_arg(arg) for arg in args] + [f"{key}={self.__str_arg(value)}" for key, value in kwargs.items()]) + ")"
                print(call_str, end="", flush=True)

            func_info = NVRTCInfo.func_signatures[self.name]
            if "dll_func" not in func_info:
                func = getattr(MetaNVRTC.dll(), self.name)
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
            error_code = nvrtcResult.NVRTC_SUCCESS
            if func_info["restype"] == c_int:
                try:
                    return_value = nvrtcResult(return_value)
                    error_code = return_value
                except:
                    pass
            elif func_info["restype"] == c_void_p:
                return_value = c_void_p(return_value)
            
            if NVRTC.print_call:
                print(f"->{return_value}", flush=True)

            if NVRTC.check_error:
                if "errcode_ret" in func_info["args"]:
                    if args_names is None:
                        args_names = list(func_info["args"].keys())

                    idx = args_names.index("errcode_ret")
                    error_code = nvrtcResult(args[idx].contents.value)

                if error_code != nvrtcResult.NVRTC_SUCCESS:
                    if error_code in NVRTCInfo.error_codes:
                        raise RuntimeError(f"{error_code}: {NVRTCInfo.error_codes[error_code]}")
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
                'c_void_p': 'void*',
                'c_char_p': 'char*',
                'LP_c_ubyte': 'uchar*'
            }
            if name in name_map:
                return name_map[name]
            else:
                return "void*"

        @staticmethod
        def __str_arg(arg)->str:
            if type(arg).__name__ == 'CArgObject':
                obj = arg._obj
                type_name = NVRTC.Func.__get_ctypes_type_name(obj)
                addr = addressof(obj)
                return f"({type_name}*)(0x{addr:x})"

            if isinstance(arg, _Pointer):
                pointed_type = arg._type_
                type_name = NVRTC.Func.__get_ctypes_type_name(pointed_type)
                addr = addressof(arg.contents) if arg else 0
                return f"({type_name}*)(0x{addr:x})"

            if isinstance(arg, Array):
                elem_type = arg._type_
                elem_type_name = NVRTC.Func.__get_ctypes_type_name(elem_type)
                length = len(arg)
                addr = addressof(arg)
                return f"({elem_type_name}[{length}])(0x{addr:x})"

            if hasattr(arg, 'value'):
                return str(arg.value)
            
            if arg is None:
                return "NULL"

            return str(arg)
        
    @staticmethod
    def init():
        try:
            for func_name, func_info in NVRTCInfo.func_signatures.items():
                if "dll_func" in func_info:
                    continue

                func = getattr(MetaNVRTC.dll(), func_name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func

            # preheat
            major = c_int(0)
            minor = c_int(0)
            ptr_major = pointer(major)
            ptr_minor = pointer(minor)
            NVRTC.nvrtcVersion(ptr_major, ptr_minor)
        except:
            pass