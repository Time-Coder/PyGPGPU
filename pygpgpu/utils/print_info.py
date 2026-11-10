# The print-info switch, and what a device reports about itself.
from __future__ import annotations
import os
from typing import List, Tuple



# Modules that own a ``print_info`` flag. Kept here, below every backend, so the
# single source of truth is importable from any layer without pulling a GPU
# runtime in: the modules are looked up in sys.modules and skipped when absent.
_PRINT_INFO_MODULES:Tuple[str, ...] = (
    "pygpgpu.opencl.driver.cl",
    "pygpgpu.cuda.driver.cuda",
    "pygpgpu.cuda.driver.nvrtc",
    "pygpgpu.hip.driver.hip",
    "pygpgpu.hip.oop.hip_compiler",
)

# The process-wide default every backend's own flag starts from. A backend module is imported
# lazily -- the call that first needs it is what pulls it in -- so flipping only the flags of
# the modules already loaded left a quiet setting at the top of a script with no effect: the
# class did not exist yet, and it then defaulted to verbose. Each module now takes its initial
# value from here, which is what makes the order of the two calls irrelevant.
_PRINT_INFO_ENABLED:bool = True


def initial_print_info() -> bool:
    """The process-wide default for a backend's own ``print_info`` flag.

    A backend calls this where it declares the flag, so a module imported *after*
    ``config.print_info = False`` starts quiet. The flag stays an ordinary class attribute
    afterwards; this only decides what it starts as.
    """
    return _PRINT_INFO_ENABLED


def nvidia_package_libraries(names: List[str]) -> List[str]:
    """Libraries shipped by the pip ``nvidia-*-cu12`` packages, newest first.

    On Linux the CUDA runtime is commonly installed with pip (``nvidia-cuda-runtime-
    cu12``, ``nvidia-cuda-nvrtc-cu12``) instead of a system toolkit. Those wheels
    drop their ``lib*.so.<N>`` files under ``site-packages/nvidia/<component>/lib``,
    which is not on the loader's search path, so a library the driver loads by bare
    SONAME is not found. This returns the matching files so a caller can load one by
    its full path.

    ``names`` are the SONAME stems ("libnvrtc", "libcudart"); nothing is returned on
    a platform where pip does not install CUDA this way.
    """
    try:
        import site
    except Exception:
        return []
    try:
        directories: List[str] = []
        try:
            directories.extend(site.getsitepackages())
        except AttributeError:                  # a virtualenv without that helper
            directories.append(os.path.dirname(os.path.dirname(os.__file__)))
        user_site = getattr(site, "getusersitepackages", None)
        if user_site is not None:
            directories.append(user_site())
    except Exception:
        return []

    found: List[str] = []
    seen: set = set()
    for directory in directories:
        component_dir: str = os.path.join(directory, "nvidia")
        if not os.path.isdir(component_dir):
            continue
        try:
            components = sorted(os.listdir(component_dir))
        except OSError:
            continue
        for component in components:
            binary_dir: str = os.path.join(component_dir, component, "lib")
            if not os.path.isdir(binary_dir):
                continue
            try:
                file_names = sorted(os.listdir(binary_dir), reverse=True)
            except OSError:
                continue
            for file_name in file_names:
                for name in names:
                    if file_name == name + ".so" or file_name.startswith(name + ".so."):
                        candidate: str = os.path.join(binary_dir, file_name)
                        if candidate not in seen:
                            seen.add(candidate)
                            found.append(candidate)
                        break
    return found


def _set_print_info(enabled:bool) -> None:
    """Turn the progress logging of every backend on or off, now and from now on.

    Each backend keeps its own ``print_info`` flag (``CL``, ``CUDA``, ``NVRTC``, ``HIP`` and
    ``HipCompiler``), all starting at ``True``, so a single kernel call prints translation,
    compilation, copy and launch lines. This flips them all in one call and only touches
    modules already in ``sys.modules``, so it never pulls in a GPU runtime just to silence
    output::

        import pygpgpu
        pygpgpu.config.print_info = False      # quiet
        pygpgpu.config.print_info = True       # verbose again (the default)

    The assignment may be made before the backends are imported, which is the usual place
    for it: it also records the value as the process-wide default, and a backend imported
    later takes its flag's initial value from there. Only flipping the loaded modules would
    have left a quiet setting at the top of a script with no effect at all, because the
    backend is imported lazily by the first call that needs it and its class did not exist
    yet -- the script then printed everything it asked to silence.

    A single backend can still be silenced on its own with its own attribute, e.g.
    ``pygpgpu.opencl.driver.CL.print_info = False``; that does not change the default for
    backends imported later, unlike this call. The math-library binders under
    ``pygpgpu/{cuda,hip}/libraries`` keep their own flag and are left alone: what they print
    is a warning about a binding that failed, not progress logging.
    """
    global _PRINT_INFO_ENABLED
    import sys

    _PRINT_INFO_ENABLED = enabled
    for module_name in _PRINT_INFO_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for value in vars(module).values():
            if isinstance(value, type) and "print_info" in vars(value):
                setattr(value, "print_info", enabled)
