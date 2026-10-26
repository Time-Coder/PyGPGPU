from __future__ import annotations

import glob
import os
from ctypes import CDLL, c_int
from typing import Callable, Dict, Optional

from .hipinfo import HIPInfo, func_signatures


def _add_dll_folder(folder:Optional[str])->None:
    """Make the folder holding the HIP runtime discoverable by the DLL loader.

    Python 3.8 added ``os.add_dll_directory`` and stopped consulting ``PATH`` for
    extension dependencies; on 3.7 the attribute does not exist at all, so that
    call raised AttributeError and HIP could never load there even with a valid
    runtime installed. 3.7 keeps using ``PATH``, which is what CPython itself
    consults before 3.8.
    """
    if not folder:
        return
    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is not None:
        # Not keeping the returned handle is deliberate: closing it would remove
        # the directory again, and the DLL stays loaded for the life of the process.
        add_dir(folder)
        return
    path = os.environ.get("PATH", "")
    if folder not in path.split(os.pathsep):
        os.environ["PATH"] = folder + os.pathsep + path


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
        candidates.append("amdhip64.dll")   # let the system loader search for it

        dll = None
        last_error:Optional[Exception]=None
        for candidate in candidates:
            try:
                folder = os.path.dirname(os.path.abspath(candidate)) if os.path.sep in candidate or "/" in candidate else None
                _add_dll_folder(folder)
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
