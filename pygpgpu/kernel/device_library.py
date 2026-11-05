"""The device-side library function registry.

CUDA's scientific-computing vocabulary is not something a user defines: the intrinsics are
built into the compiler, and the library functions (cuBLAS, cuSOLVER, cuFFT, ...) are
host-side APIs. What a user *can* do is point at a device-side function that already
exists -- in a header they ship, or in one of CUDA's own headers -- and use it in a kernel
body with a name of their own choosing.

That is what this module is for:

    from pygpgpu import device_function

    @device_function(cuda="__my_fast_rcp", opencl="native_recip",
                     headers={"cuda": "myclib.h"})
    def my_fast_rcp(x: float) -> float:
        \"\"\"The body is the host's; the device gets the symbol the table names.\"\"\"

    @kernel
    def k(x: np.ndarray, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = my_fast_rcp(x[i])

The body is never read by the translator: the device gets the symbol the table names, and
the host (the ``python`` backend) runs the body. The two therefore have to agree, which is
the same rule the builtins follow -- and the reason the registry is keyed by the *object*
the callee resolves to, not by the spelling of the name.

**What a registered function can and cannot be.** The device call is the named symbol with
the arguments as they are, so the symbol must be a device-callable function of that name:
a compiler intrinsic, or a ``__device__`` function in a header the compilation includes.
A *host* library function (a cuBLAS call, say) cannot be registered this way -- it takes
pointers and returns a status, and the device has no such thing. Those live in the
host-side bindings (``pygpgpu.cuda.libraries``) instead, and they are array-level
operations, not per-work-item ones.

The headers are the other half of it: a device function that lives in a header the
compilation does not include is not callable, so the registry records what the compilation
needs and the generated source includes it.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Set, Union

__all__ = ["device_function", "device_symbol", "library_headers_for",
           "library_functions", "library_function_name"]


# The registry. Keyed by the *object* the callee resolves to, so the translator recognizes
# a registered function by identity rather than by the spelling of its name -- the same
# rule the builtins follow, and for the same reason.
_BY_FUNCTION:Dict[object, "_DeviceFunction"] = {}
_BY_NAME:Dict[str, "_DeviceFunction"] = {}


class _DeviceFunction:
    """One registered device-side function."""

    def __init__(self, name:str, symbols:Dict[str, str], headers:Dict[str, List[str]],
                 function:Callable):
        self.name:str = name
        # backend -> the device symbol to call. A backend with no entry cannot run it.
        self.symbols:Dict[str, str] = symbols
        # backend -> the headers the compilation needs for it.
        self.headers:Dict[str, List[str]] = headers
        self.function:Callable = function


def device_function(function:Optional[Callable]=None, *,
                    cuda:Optional[str]=None,
                    opencl:Optional[str]=None,
                    hip:Optional[str]=None,
                    headers:Optional[Dict[str, Union[str, List[str]]]]=None,
                    name:Optional[str]=None)->Callable:
    """Register a function as a device-side library function.

    ``cuda`` / ``opencl`` / ``hip`` name the symbol the device calls on that backend. A
    backend with no symbol cannot run the function, and the translator says so by name.

    ``headers`` maps a backend to what the compilation needs for it. Each entry is either

    * a **path to a header file** -- the package copies it into the cache's ``include/``
      folder and the generated source includes it by name, so the user does not have to
      know where the cache is; or
    * a **full ``#include`` line** -- the package carries it into the generated source
      verbatim, which is what a system header needs (``#include <cuda_fp16.h>``).

    A list of either is accepted too.

    The decorated function's body is the *host's* implementation (the ``python`` backend
    runs it); the device gets the symbol named above.
    """
    def decorate(target:Callable)->Callable:
        symbols:Dict[str, str] = {}
        if cuda is not None:
            symbols["cuda"] = cuda
        if opencl is not None:
            symbols["opencl"] = opencl
        if hip is not None:
            symbols["hip"] = hip
        if not symbols:
            raise ValueError(
                f"device_function() needs at least one backend symbol: name the device "
                f"function for cuda, opencl or hip (got none for '{name or target.__name__}')")

        resolved_headers:Dict[str, List[str]] = {}
        if headers:
            for backend, value in headers.items():
                if isinstance(value, str):
                    resolved_headers[backend] = [value]
                else:
                    resolved_headers[backend] = list(value)

        registered_name:str = name or target.__name__
        entry = _DeviceFunction(registered_name, symbols, resolved_headers, target)
        _BY_FUNCTION[target] = entry
        _BY_NAME[registered_name] = entry
        return target

    if function is not None:
        return decorate(function)
    return decorate


def device_symbol(target:object, backend:str)->Optional[str]:
    """The device symbol a registered function has on ``backend``, or None.

    None means the callee is not a registered device function at all, or it has no
    implementation on that backend.
    """
    entry:Optional[_DeviceFunction] = _BY_FUNCTION.get(target)
    if entry is None:
        return None
    return entry.symbols.get(backend)


def library_function_name(target:object)->Optional[str]:
    """The registered name of a callee, or None when it is not a registered one."""
    entry:Optional[_DeviceFunction] = _BY_FUNCTION.get(target)
    return entry.name if entry is not None else None


def library_functions()->Dict[str, _DeviceFunction]:
    """Every registered function, by name."""
    return dict(_BY_NAME)


def library_headers_for(backend:str, functions:Iterable[Union[object, str]]=())->Set[str]:
    """The headers the given functions need on ``backend``.

    ``functions`` is any iterable of registered function *objects* (or of their names).
    """
    headers:Set[str] = set()
    for item in functions:
        entry:Optional[_DeviceFunction] = None
        if isinstance(item, str):
            entry = _BY_NAME.get(item)
        else:
            entry = _BY_FUNCTION.get(item)
        if entry is None:
            continue
        headers.update(entry.headers.get(backend, ()))
    return headers


# The folder a cache keeps its device headers in, beside the sources that include them.
INCLUDE_FOLDER:str = "include"


def _install_header(source:str, directory:str)->str:
    """Put a header file into the cache's include folder and return its include line.

    A quoted ``#include`` resolves relative to the file that writes it, so the header has
    to sit beside the generated source -- in the cache's own ``include/`` folder, which is
    where the package's own header (the quaternion one) goes too.
    """
    import os
    import shutil

    if not os.path.isfile(source):
        raise FileNotFoundError(
            f"the header '{source}' a device function asked for does not exist")
    target_folder:str = os.path.join(directory, INCLUDE_FOLDER)
    os.makedirs(target_folder, exist_ok=True)
    name:str = os.path.basename(source)
    target:str = os.path.join(target_folder, name)
    # The copy is refreshed whenever the source changed, so editing the header and
    # re-running a kernel picks the new text up.
    if os.path.isfile(target):
        with open(target, "r", encoding="utf-8") as file:
            existing:str = file.read()
        with open(source, "r", encoding="utf-8") as file:
            current:str = file.read()
        if existing == current:
            return f'#include "{INCLUDE_FOLDER}/{name}"'
    shutil.copyfile(source, target)
    return f'#include "{INCLUDE_FOLDER}/{name}"'


def include_lines_for(backend:str, headers:Iterable[str], directory:str)->List[str]:
    """The ``#include`` lines the given headers need, as the generated source spells them.

    A header given as a **path** is a file of the user's own: it is copied into the
    cache's include folder and included by name. A header given as a **full include
    line** is carried into the generated source verbatim -- that is what a system header
    needs, since the compiler finds it on its own search path.
    """
    if headers is None:
        return []
    entries:List[str] = [headers] if isinstance(headers, str) else list(headers)
    lines:List[str] = []
    for entry in entries:
        if entry.lstrip().startswith("#include"):
            lines.append(entry.strip())
        else:
            lines.append(_install_header(entry, directory))
    return lines
