from __future__ import annotations
import platform
from typing import Dict, Callable
from ctypes import c_int, c_void_p, CDLL, _Pointer, Array, addressof

from .cuinfo import CUInfo
from .cutypes import CUresult


class MetaCUDA(type):

    __dll = None
    __func_map:Dict[str, Callable] = {}

    @staticmethod
    def cuda_lib_path():
        system = platform.system()
        if system == "Windows":
            lib_name = "nvcuda.dll"
        elif system == "Linux":
            lib_name = "libcuda.so"
        elif system == "Darwin":  # macOS
            lib_name = "libcuda.dylib"
        else:
            raise RuntimeError(f"unsupported operating system: {system}")
        
        return lib_name
    
    @staticmethod
    def dll():
        if MetaCUDA.__dll is None:
            MetaCUDA.__dll = CDLL(MetaCUDA.cuda_lib_path())

        return MetaCUDA.__dll

    def __getattr__(self, name:str)->Callable:
        if name not in MetaCUDA.__func_map:
            MetaCUDA.__func_map[name] = self.Func(name)

        return MetaCUDA.__func_map[name]


class CUDA(metaclass=MetaCUDA):

    print_call:bool = False
    print_info:bool = True
    check_error:bool = True

    class Func:

        def __init__(self, name:str):
            self.name = name

        def __call__(self, *args, **kwargs):
            if CUDA.print_call:
                call_str = self.name + "(" + ", ".join([self.__str_arg(arg) for arg in args] + [f"{key}={self.__str_arg(value)}" for key, value in kwargs.items()]) + ")"
                print(call_str, end="", flush=True)

            func_info = CUInfo.func_signatures[self.name]
            if "dll_func" not in func_info:
                func = getattr(MetaCUDA.dll(), self.name)
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
            error_code = CUresult.CUDA_SUCCESS
            if func_info["restype"] == c_int:
                try:
                    return_value = CUresult(return_value)
                    error_code = return_value
                except:
                    pass
            elif func_info["restype"] == c_void_p:
                return_value = c_void_p(return_value)
            
            if CUDA.print_call:
                print(f"->{return_value}", flush=True)

            if CUDA.check_error:
                if "errcode_ret" in func_info["args"]:
                    if args_names is None:
                        args_names = list(func_info["args"].keys())

                    idx = args_names.index("errcode_ret")
                    error_code = CUresult(args[idx].contents.value)

                if error_code != CUresult.CUDA_SUCCESS:
                    if error_code in CUInfo.error_codes:
                        raise RuntimeError(f"{error_code}: {CUInfo.error_codes[error_code]}")
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
                type_name = CUDA.Func.__get_ctypes_type_name(obj)
                addr = addressof(obj)
                return f"({type_name}*)(0x{addr:x})"

            if isinstance(arg, _Pointer):
                pointed_type = arg._type_
                type_name = CUDA.Func.__get_ctypes_type_name(pointed_type)
                addr = addressof(arg.contents) if arg else 0
                return f"({type_name}*)(0x{addr:x})"

            if isinstance(arg, Array):
                elem_type = arg._type_
                elem_type_name = CUDA.Func.__get_ctypes_type_name(elem_type)
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
            for func_name, func_info in CUInfo.func_signatures.items():
                if "dll_func" in func_info:
                    continue

                func = getattr(MetaCUDA.dll(), func_name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func

            # preheat
            CUDA.cuInit(0)
        except:
            pass