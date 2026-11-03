from __future__ import annotations
import os
import platform
import threading
from typing import Dict, Callable, Optional
from ctypes import c_int, c_void_p, c_uint, pointer, addressof, _Pointer, Array, CDLL

from .clinfo import CLInfo
from .cltypes import ErrorCode

# Same background-initialization contract as the CUDA driver (see pygpgpu/cuda/driver/cuda.py):
# the runtime is initialized off the import path, and `Func.__call__` waits for it. Both
# functions are private -- the backend package starts the thread when it is imported, and a
# caller never asks for the initialization or waits for it explicitly.
_INIT_DONE = threading.Event()
_INIT_THREAD:Optional[threading.Thread] = None
_INIT_LOCK = threading.Lock()


def _start_init()->None:
    """Run `init()` once, on a background thread (idempotent)."""
    global _INIT_THREAD
    with _INIT_LOCK:
        if _INIT_THREAD is not None:
            return

        def _run()->None:
            try:
                CL.init()
            finally:
                _INIT_DONE.set()

        _INIT_THREAD = threading.Thread(target=_run, name="pygpgpu-opencl-init")
        _INIT_THREAD.start()


def _wait_ready()->None:
    """Block until the background runtime initialization has finished (or failed)."""
    _INIT_DONE.wait()


class MetaCL(type):

    __dll = None
    __func_map:Dict[str, Callable] = {}
    # The exception init() hit when the machine has no runtime (usually the CDLL OSError
    # for OpenCL.dll). Diagnostic only: every real call re-raises what the driver itself
    # raises, so nothing here can make a broken runtime look usable.
    last_init_error:Optional[BaseException] = None

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
            MetaCL.__dll = CDLL(MetaCL.opencl_lib_path())

        return MetaCL.__dll

    def __getattr__(self, name:str)->Callable:
        if name not in MetaCL.__func_map:
            MetaCL.__func_map[name] = self.Func(name)

        return MetaCL.__func_map[name]


class CL(metaclass=MetaCL):

    print_call:bool = False
    print_info:bool = True
    check_error:bool = True

    class Func:

        def __init__(self, name:str):
            self.name = name

        def __call__(self, *args, **kwargs):
            # The one synchronization point with the background initialization (see cuda.py).
            _wait_ready()

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
            elif func_info["restype"] == c_void_p:
                return_value = c_void_p(return_value)
            
            if CL.print_call:
                print(f"->{return_value}", flush=True)

            if CL.check_error:
                if "errcode_ret" in func_info["args"]:
                    if args_names is None:
                        args_names = list(func_info["args"].keys())

                    idx = args_names.index("errcode_ret")
                    error_code = ErrorCode(args[idx].contents.value)

                if error_code != ErrorCode.CL_SUCCESS:
                    if error_code in func_info["errors"]:
                        raise RuntimeError(f"{error_code}: {func_info['errors'][error_code]}")
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
                type_name = CL.Func.__get_ctypes_type_name(obj)
                addr = addressof(obj)
                return f"({type_name}*)(0x{addr:x})"

            if isinstance(arg, _Pointer):
                pointed_type = arg._type_
                type_name = CL.Func.__get_ctypes_type_name(pointed_type)
                addr = addressof(arg.contents) if arg else 0
                return f"({type_name}*)(0x{addr:x})"

            if isinstance(arg, Array):
                elem_type = arg._type_
                elem_type_name = CL.Func.__get_ctypes_type_name(elem_type)
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
        """Pre-bind the driver's entry points, once, at import time.

        A runtime the machine does not have must not fail the import: the entries are
        bound lazily anyway (`Func.__call__` loads them on first use), so a missing DLL
        here only means "nothing to pre-bind". The error is recorded for the diagnostic
        and every real call then re-raises what the driver itself raises -- nothing can
        make a broken runtime look usable.
        """
        try:
            for func_name, func_info in CLInfo.func_signatures.items():
                if "dll_func" in func_info:
                    continue

                func = getattr(MetaCL.dll(), func_name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func

            # preheat -- through the raw dll, NOT through Func.__call__: init itself runs
            # on the background thread, and Func waits for init, so going through Func
            # here would wait for itself.
            n_platforms = c_uint()
            ptr_n_platforms = pointer(n_platforms)
            MetaCL.dll().clGetPlatformIDs(0, None, ptr_n_platforms)
        except BaseException as error:
            MetaCL.last_init_error = error