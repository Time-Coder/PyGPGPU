from __future__ import annotations
import os
import platform
import threading

from typing import Dict, Callable, Optional
from ctypes import c_int, c_void_p, CDLL, _Pointer, Array, addressof
from ...utils import initial_print_info

from .cuinfo import CUInfo
from .cutypes import CUresult

# The driver is initialized on a background thread started at import, so `import
# pygpgpu.cuda` does not block on DLL load + cuInit (which is a GIL-releasing C call and
# overlaps with whatever the importing thread does next). Every driver call goes through
# `Func.__call__`, which waits for that initialization to finish before touching the
# driver -- the wait is the whole synchronization story, and nothing else needs to know.
#
# Both functions are private: the backend package starts the thread when it is imported,
# and a caller never asks for the initialization or waits for it explicitly.
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
                CUDA.init()
            finally:
                _INIT_DONE.set()

        _INIT_THREAD = threading.Thread(target=_run, name="pygpgpu-cuda-init")
        _INIT_THREAD.start()


def _wait_ready()->None:
    """Block until the background driver initialization has finished (or failed)."""
    _INIT_DONE.wait()


class MetaCUDA(type):

    __dll = None
    __func_map:Dict[str, Callable] = {}
    # The exception init() hit when the machine has no runtime (usually the CDLL OSError
    # for nvcuda.dll). None once a call has actually succeeded. Diagnostic only: every
    # real call re-raises whatever the driver itself raises, so nothing here can make a
    # broken runtime look usable.
    last_init_error:Optional[BaseException] = None

    @staticmethod
    def cuda_lib_path():
        system = platform.system()
        if system == "Windows":
            lib_name = "nvcuda.dll"
        elif system == "Linux":
            # The unversioned SONAME usually only exists in the driver's dev package;
            # the versioned one is the file the loader actually keeps, so try both.
            lib_name = "libcuda.so"
            if not os.path.isfile(lib_name):
                lib_name = "libcuda.so.1"
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
    print_info:bool = initial_print_info()
    check_error:bool = True

    class Func:

        def __init__(self, name:str):
            self.name = name

        def __call__(self, *args, **kwargs):
            # The one synchronization point with the background initialization: a call
            # made before it finished simply waits for it, then runs (or fails with the
            # driver's own error, if the runtime is missing).
            _wait_ready()

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

            if CUDA.check_error and self.name not in CUInfo.no_need_check_error_funcs:
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
        # A runtime the machine does not have must not fail the import: the entries are
        # bound lazily anyway (Func.__call__ loads them on first use), so a missing DLL
        # here only means "nothing to pre-bind". The error is recorded for the diagnostic
        # and the next real driver call fails with the same OSError it always raised.
        try:
            for func_name, func_info in CUInfo.func_signatures.items():
                if "dll_func" in func_info:
                    continue

                func = getattr(MetaCUDA.dll(), func_name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func
        except BaseException as error:
            MetaCUDA.last_init_error = error
            if CUDA.print_info:
                print(f"warning: could not pre-bind the CUDA driver "
                      f"({type(error).__name__}: {error}); "
                      f"calls will fail when one is made.")

        # preheat -- through the raw dll, NOT through Func.__call__: init itself runs on
        # the background thread, and Func waits for init, so going through Func here
        # would wait for itself.
        try:
            MetaCUDA.dll().cuInit(0)
        except BaseException as error:
            MetaCUDA.last_init_error = error