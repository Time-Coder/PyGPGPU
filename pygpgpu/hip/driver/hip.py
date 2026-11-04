from __future__ import annotations

import glob
import os
import platform
import threading
from ctypes import CDLL, c_int
from typing import Callable, Dict, List, Optional

from .hipinfo import HIPInfo, func_signatures

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
                HIP.init()
            finally:
                _INIT_DONE.set()

        _INIT_THREAD = threading.Thread(target=_run, name="pygpgpu-hip-init")
        _INIT_THREAD.start()


def _wait_ready()->None:
    """Block until the background runtime initialization has finished (or failed)."""
    _INIT_DONE.wait()


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
    # The exception init() hit when the machine has no runtime (usually the RuntimeError
    # from loading amdhip64). Diagnostic only: every real call re-raises what the driver
    # itself raises, so nothing here can make a broken runtime look usable.
    last_init_error:Optional[BaseException] = None

    @staticmethod
    def rocm_roots():
        # On Linux the standard install root is /opt/rocm (usually a symlink to the
        # versioned folder), and ROCM_PATH points at it when the standard layout is
        # not used. The version folder is folded into the root the same way the
        # Windows scan folds it, so "root/bin/clang" stays the layout every caller
        # assumes.
        if platform.system() != "Windows":
            roots = []
            env = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
            if env:
                roots.append(env)
            roots.extend(glob.glob("/opt/rocm*"))
            roots.append("/usr")   # distro-packaged ROCm keeps its runtime in /usr/lib
            expanded:List[str] = []
            for root in roots:
                if not os.path.isdir(root):
                    continue
                versions = sorted(glob.glob(f"{root}/*"))
                if versions and any(os.path.isfile(os.path.join(v, "bin", "clang"))
                                    for v in versions):
                    expanded.extend(versions)
                else:
                    expanded.append(root)
            return expanded

        roots = []
        env = os.environ.get("HIP_PATH")
        if env:
            roots.append(env.replace("\\", "/"))
        # The installer's own default is "C:\Program Files\AMD\ROCm" (one version folder
        # under it), and a zip install usually lands on a drive root. Both are looked at;
        # the version folder is folded into the root so "root/bin/clang.exe" stays the
        # layout every caller assumes.
        for drive in ("C:", "D:", "E:"):
            roots.extend(glob.glob(f"{drive}/AMD/ROCm*"))
            roots.extend(glob.glob(f"{drive}/Program Files/AMD/ROCm"))
            roots.extend(glob.glob(f"{drive}/Program Files (x86)/AMD/ROCm"))
        expanded:List[str] = []
        for root in roots:
            versions = sorted(glob.glob(f"{root}/*"))
            if versions and any(os.path.isfile(os.path.join(v, "bin", "clang.exe")) for v in versions):
                expanded.extend(versions)
            else:
                expanded.append(root)
        return expanded

    @classmethod
    def dll(cls):
        if MetaHIP.__dll is not None:
            return MetaHIP.__dll

        if platform.system() != "Windows":
            # Linux: the runtime is libamdhip64.so (a plain SONAME the loader finds
            # through the standard library paths or ROCm's own ld.so configuration),
            # with /usr/lib as the fallback for a distro package that ships it there.
            candidates = ["libamdhip64.so", "libhip_hcc.so"]
            for root in cls.rocm_roots():
                candidates.extend(sorted(glob.glob(f"{root}/lib/libamdhip64.so*"), reverse=True))
            dll = None
            last_error:Optional[Exception] = None
            for candidate in candidates:
                try:
                    dll = CDLL(candidate)
                    break
                except OSError as e:
                    last_error = e
            if dll is None:
                # The loader's own error type: every other backend surfaces a
                # missing runtime as OSError, and callers (the probe, kernel_import,
                # the examples' backend fallback) are written against that.
                raise OSError(f"cannot load libamdhip64.so (HIP runtime not found); "
                              f"roots={cls.rocm_roots()}: {last_error}") from last_error
            MetaHIP.__dll = dll
            return dll

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
            # The one synchronization point with the background initialization (see cuda.py).
            _wait_ready()

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
        """Pre-bind the driver's entry points, once, at import time.

        A runtime the machine does not have must not fail the import: the entries are
        bound lazily anyway (`Func.__call__` loads them on first use), so a missing DLL
        here only means "nothing to pre-bind". The error is recorded for the diagnostic
        and every real call then re-raises what the driver itself raises -- nothing can
        make a broken runtime look usable.
        """
        try:
            for func_name, func_info in func_signatures.items():
                if "dll_func" in func_info:
                    continue

                func = getattr(MetaHIP.dll(), func_name)
                func.argtypes = list(func_info["args"].values())
                func.restype = func_info["restype"]
                func_info["dll_func"] = func
        except BaseException as error:
            MetaHIP.last_init_error = error
            if HIP.print_info:
                print(f"warning: could not pre-bind the HIP runtime "
                      f"({type(error).__name__}: {error}); "
                      f"calls will fail when one is made.")

        # preheat -- through the raw dll, NOT through Func.__call__: init itself runs on
        # the background thread, and Func waits for init, so going through Func here
        # would wait for itself.
        try:
            MetaHIP.dll().hipInit(0)
        except BaseException as error:
            MetaHIP.last_init_error = error
