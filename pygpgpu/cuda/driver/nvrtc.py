from __future__ import annotations
import platform
import shutil
import glob
import os
import threading
from typing import Dict, Callable, List, Optional
from ctypes import c_int, c_void_p, pointer, CDLL, addressof, _Pointer, Array

from .nvrtcinfo import NVRTCInfo
from .nvrtctypes import nvrtcResult

# Same background-initialization contract as the CUDA driver (see cuda.py): the compiler
# is initialized off the import path, and `Func.__call__` waits for it. Both functions are
# private -- the backend package starts the thread when it is imported, and a caller never
# asks for the initialization or waits for it explicitly.
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
                NVRTC.init()
            finally:
                _INIT_DONE.set()

        _INIT_THREAD = threading.Thread(target=_run, name="pygpgpu-nvrtc-init")
        _INIT_THREAD.start()


def _wait_ready()->None:
    """Block until the background compiler initialization has finished (or failed)."""
    _INIT_DONE.wait()


class MetaNVRTC(type):

    __dll = None
    __func_map:Dict[str, Callable] = {}
    # The exception init() hit when the machine has no CUDA toolkit (usually the
    # FileNotFoundError for nvcc.exe / the nvrtc dll). Diagnostic only: every real call
    # re-raises what the loader itself raises, so nothing here can make a broken setup
    # look usable.
    last_init_error:Optional[BaseException] = None

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
            # The toolkit's runtime package ships a versioned SONAME
            # (libnvrtc.so.12, libnvrtc.so.13 ...); the unversioned name needs the
            # dev package, so the versioned ones are probed first. /usr/local/cuda
            # is scanned explicitly because its lib folder is not always on the
            # loader's default search path, and the pip nvidia-cuda-nvrtc-cu12
            # package keeps its copies under site-packages/nvidia/cuda_nvrtc/lib,
            # which no loader looks at either -- the full path is what finds them.
            import glob as _glob

            from ...utils import nvidia_package_libraries

            candidates:List[str] = sorted(_glob.glob("/usr/local/cuda/lib64/libnvrtc.so*"), reverse=True)
            candidates.extend(sorted(_glob.glob("/usr/local/cuda/lib/libnvrtc.so*"), reverse=True))
            # A pip install is the common Linux arrangement, so its exact path is
            # preferred over a bare name the loader may not resolve.
            candidates = nvidia_package_libraries(["libnvrtc"]) + candidates
            candidates.extend(["libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11"])
            lib_name = candidates[-1]
            for candidate in candidates:
                if "/" in candidate:
                    if os.path.isfile(candidate):
                        lib_name = candidate
                        break
                else:
                    lib_name = candidate
                    break
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
            # The one synchronization point with the background initialization (see cuda.py).
            _wait_ready()

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
                    raise RuntimeError(f"{error_code}")

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
        """Pre-bind the compiler's entry points, once, at import time.

        A toolkit the machine does not have must not fail the import: the entries are
        bound lazily anyway (`Func.__call__` loads them on first use), so a missing DLL
        here only means "nothing to pre-bind". The error is recorded for the diagnostic
        and every real call then re-raises what the loader itself raises -- nothing can
        make a broken setup look usable.
        """
        try:
            for func_name, func_info in NVRTCInfo.func_signatures.items():
                if "dll_func" in func_info:
                    continue

                func = getattr(MetaNVRTC.dll(), func_name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func
        except BaseException as error:
            MetaNVRTC.last_init_error = error
            if NVRTC.print_info:
                print(f"warning: could not pre-bind NVRTC "
                      f"({type(error).__name__}: {error}); "
                      f"calls will fail when one is made.")

        # preheat -- through the raw dll, NOT through Func.__call__: init itself runs on
        # the background thread, and Func waits for init, so going through Func here
        # would wait for itself.
        try:
            major = c_int(0)
            minor = c_int(0)
            ptr_major = pointer(major)
            ptr_minor = pointer(minor)
            MetaNVRTC.dll().nvrtcVersion(ptr_major, ptr_minor)
        except BaseException as error:
            MetaNVRTC.last_init_error = error