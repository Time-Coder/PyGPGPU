from __future__ import annotations

import glob
import os
import platform
from ctypes import CDLL, c_int, c_char_p
from typing import Callable, Dict

from .hipinfo import HIPInfo, func_signatures


class MetaHIP(type):

    __dll = None
    __func_map:Dict[str, Callable] = {}

    @staticmethod
    def rocm_roots():
        roots = []
        env = os.environ.get("HIP_PATH")
        if env:
            roots.append(env.replace("\\", "/"))
        for drive in ("C:", "D:", "E:"):
            roots.extend(glob.glob(f"{drive}/AMD/ROCm*"))
        return roots

    @classmethod
    def dll(cls):
        if MetaHIP.__dll is not None:
            return MetaHIP.__dll

        candidates = []
        for root in cls.rocm_roots():
            candidates.extend(sorted(glob.glob(f"{root}/bin/amdhip64*.dll"), reverse=True))
        candidates.append("amdhip64.dll")   # 交给系统搜索

        dll = None
        last_error:Exception = None
        for candidate in candidates:
            try:
                folder = os.path.dirname(os.path.abspath(candidate)) if os.path.sep in candidate or "/" in candidate else None
                if folder:
                    os.add_dll_directory(folder)
                dll = CDLL(candidate)
                break
            except OSError as e:
                last_error = e

        if dll is None:
            raise RuntimeError(f"cannot load amdhip64 dll (HIP runtime not found); roots={cls.rocm_roots()}: {last_error}")

        MetaHIP.__dll = dll
        return dll

    def __getattr__(self, name:str)->Callable:
        if name not in MetaHIP.__func_map:
            MetaHIP.__func_map[name] = self.Func(name)

        return MetaHIP.__func_map[name]


class HIP(metaclass=MetaHIP):

    print_call:bool = False
    print_info:bool = True
    check_error:bool = True

    class Func:

        def __init__(self, name:str):
            self.name:str = name

        def __call__(self, *args, **kwargs):
            if HIP.print_call:
                print(self.name + "(...)", end="", flush=True)

            func_info = func_signatures[self.name]
            if "dll_func" not in func_info:
                func = getattr(MetaHIP.dll(), self.name)
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
            error_code:int = return_value if func_info["restype"] == c_int else 0

            if HIP.print_call:
                print(f"->{return_value}", flush=True)

            if HIP.check_error and self.name not in HIPInfo.no_need_check_error_funcs:
                if error_code != 0:
                    try:
                        message = HIP.hipGetErrorString(error_code)
                        message = message.decode("utf-8", errors="replace") if isinstance(message, bytes) else str(message)
                    except Exception:
                        message = "unknown error"
                    raise RuntimeError(f"hipError {error_code}: {message} ({self.name})")

            return return_value

    @staticmethod
    def init():
        for func_name, func_info in func_signatures.items():
            if "dll_func" in func_info:
                continue

            try:
                func = getattr(MetaHIP.dll(), func_name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func
            except (TypeError, AttributeError):
                if HIP.print_info:
                    print(f"warning: failed to load {func_name} from amdhip64, skipped.")

        try:
            HIP.hipInit(0)
        except Exception:
            pass
