from __future__ import annotations

import asyncio
import ast
import concurrent.futures
import functools
import inspect
import os
import pathlib
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from ..vec_types.layout import pack
from ..utils import local_size_for_global, detect_work_size, device_limits
from .python_backend import PythonKernel
from . import structs
from .translator import (
    FunctionTranslator,
    infer_ctype,
    is_ptr,
    scan_self_attrs,
    short_hash,
    type_sig,
)
from .objects import ObjectBinding, convert_object_arguments


# ---------- bracket selector parsing ----------

_BACKEND_ALIASES:Dict[str, str] = {
    "opencl": "opencl", "ocl": "opencl", "cl": "opencl",
    "cuda": "cuda", "cu": "cuda",
    "hip": "hip", "rocm": "hip", "amd": "hip",
    # The pure-python backend: no translation, no device, serial execution on the host.
    "python": "python", "py": "python", "host": "python",
}

# Probe order when no backend is specified: take the first backend that enumerates at least one device
DEFAULT_BACKEND_ORDER:Tuple[str, ...] = ("cuda", "hip", "opencl")

_probed_backend:Optional[str] = None

# The selector list a bracketed-less call walks, set by set_default_device(). None means
# "use DEFAULT_BACKEND_ORDER". Each entry is (backend, device-or-None).
_default_devices:Optional[List[Tuple[Optional[str], Optional[str]]]] = None
_default_device_lock = threading.Lock()


def set_default_device(*selectors:str)->None:
    """Choose what a kernel call without a bracket runs on, in order of preference.

    ``set_default_device("opencl:cpu", "cuda")`` makes a plain ``k(...)`` try the CPU
    OpenCL device first and fall back to cuda when that is unusable. Entries are the same
    selector strings the brackets accept -- ``"backend"``, ``"backend:device"`` or a bare
    device name -- and each is tried until one resolves.

    Called with no arguments it restores the built-in order (``cuda -> hip -> opencl``).

    A kernel decorated with an explicit ``@kernel(backend=...)`` keeps that backend and is
    unaffected; this only decides what an unspecified call means. The list is read at call
    time, so it applies to kernels that were already decorated.
    """
    global _default_devices
    if not selectors:
        with _default_device_lock:
            _default_devices = None
        return

    parsed:List[Tuple[Optional[str], Optional[str]]] = []
    for selector in selectors:
        if not isinstance(selector, str):
            raise TypeError(
                "set_default_device() takes selector strings like 'opencl:cpu', got %s"
                % type(selector).__name__)
        backend, device = _parse_selector(selector)
        if backend is None and device is None:
            raise ValueError("empty device selector")
        parsed.append((backend, device))

    with _default_device_lock:
        _default_devices = parsed


def get_default_device()->List[str]:
    """The selector list an unspecified kernel call currently walks."""
    with _default_device_lock:
        entries = _default_devices
    if entries is None:
        return list(DEFAULT_BACKEND_ORDER)
    return [("%s:%s" % (backend, device)) if backend and device
            else (backend or device or "")
            for backend, device in entries]


def _entry_usable(backend:Optional[str], device:Optional[str])->Optional[Tuple[str, Optional[str]]]:
    """Resolve one default entry to a (backend, device) pair, or None if unusable.

    A backend-less entry (a bare device name) is looked up across the standard order, so
    ``set_default_device("gfx1036")`` finds it whichever backend owns it.
    """
    candidates:List[str] = [backend] if backend is not None else list(DEFAULT_BACKEND_ORDER)
    for candidate in candidates:
        if _probe_backend(candidate) is None:
            continue
        if device is None:
            return candidate, None
        try:
            _resolve_device(candidate, device)
        except Exception:
            continue
        return candidate, device
    return None


def _default_target()->Tuple[str, Optional[str]]:
    """The backend (and device) a call without a bracket should use.

    Walks the configured list first, then the built-in order; the result is cached only
    while the configuration is unchanged, because set_default_device() may change it at
    any time.
    """
    global _probed_backend
    with _default_device_lock:
        entries = _default_devices
    if entries is not None:
        for backend, device in entries:
            resolved = _entry_usable(backend, device)
            if resolved is not None:
                return resolved
        raise RuntimeError(
            "none of the devices set by set_default_device() is usable: %s. "
            "Check the names against list_devices(), or call set_default_device() with "
            "no arguments to restore the built-in order."
            % ", ".join(get_default_device()))

    if _probed_backend is None:
        for backend in DEFAULT_BACKEND_ORDER:
            if _probe_backend(backend) is not None:
                _probed_backend = backend
                break
        else:
            _probed_backend = "opencl"
    return _probed_backend, None


def _backend_devices(backend:str)->List[Any]:
    """Enumerate the backend's device list (order = numbering used by plain-numeric selectors); raises when the runtime is missing."""
    if backend == "opencl":
        from ..opencl.oop.platforms import Platforms
        return [d for platform in Platforms for d in platform.devices]
    if backend == "hip":
        from ..hip.oop.devices import Devices
        return list(Devices)
    if backend == "cuda":
        from ..cuda.oop.devices import Devices
        return list(Devices)
    raise ValueError(f"unknown backend '{backend}'")


def _probe_backend(backend:str)->Optional[str]:
    """Probe whether a backend is usable: enumerating at least one device counts as usable.

    Backend packages load their runtime (nvcuda/amdhip64/OpenCL) only at import time; failures are treated as unavailable.
    """
    try:
        if _backend_devices(backend):
            return backend
    except Exception:
        pass
    return None


def _default_backend()->str:
    """The backend an unspecified call uses, honouring set_default_device()."""
    return _default_target()[0]


def _parse_selector(sel:str)->Tuple[Optional[str], Optional[str]]:
    """Parse selectors of the form "backend" / "backend:device" / "device".

    Returns (backend, device); unspecified parts are None (filled in by defaults / earlier brackets).
    """
    if not isinstance(sel, str):
        raise TypeError(f"kernel bracket selector must be a string like 'opencl:gfx1036', got {type(sel).__name__}")
    sel = sel.strip()
    if not sel:
        raise ValueError("empty kernel bracket selector")
    if ":" in sel:
        head, rest = sel.split(":", 1)
        backend:Optional[str] = _BACKEND_ALIASES.get(head.strip().lower())
        if backend is not None:
            return backend, (rest.strip() or None)
        # the part after the colon is not a backend name -> the whole string is a device identifier (e.g. "geforce:0")
        return None, sel
    backend = _BACKEND_ALIASES.get(sel.lower())
    if backend is not None:
        return backend, None
    return None, sel


def _parse_work_item(item:Any)->Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """Parse a work-size bracket: int / (global,) / (global, local); None means automatic."""
    def norm(v:Any)->Optional[Tuple[int, ...]]:
        if v is None:
            return None
        if isinstance(v, int):
            return (v,)
        if isinstance(v, tuple):
            t:Tuple[int, ...] = tuple(int(x) for x in v)
            if not t:
                raise ValueError("work size tuple cannot be empty")
            return t
        raise TypeError(f"work size entry must be int or tuple of ints, got {type(v).__name__}")

    if isinstance(item, int):
        return norm(item), None
    if isinstance(item, tuple):
        if len(item) == 1:
            return norm(item[0]), None
        if len(item) == 2:
            return norm(item[0]), norm(item[1])
    raise TypeError("work size bracket must be n, (global,), or (global, local); use None for auto")


# Vendor spellings people reach for, mapped to a substring of the vendor string the
# drivers actually report. "amd" is the common case: the OpenCL vendor string is
# "Advanced Micro Devices, Inc.", which does not contain the abbreviation, so a plain
# substring match for "amd" would never find an AMD device.
_VENDOR_ALIASES:Dict[str, str] = {
    "amd": "advanced micro devices",
    "nvidia": "nvidia",
    "intel": "intel",
    "apple": "apple",
    "qualcomm": "qualcomm",
    "arm": "arm",
    "pocl": "pocl",
}


def _device_match_keys(sel:str, device:Any)->bool:
    """Does this device answer to the selector?

    A selector matches by substring against the device's name, vendor and arch, plus a
    small table of vendor aliases so a short, commonly written vendor name works even
    when the driver reports the full company name.
    """
    low:str = sel.lower()
    vendor:str = (getattr(device, "vendor", "") or "").lower()
    arch:str = (getattr(device, "arch", "") or "").lower()

    if low in device.name.lower() or low in vendor or low in arch:
        return True

    alias:Optional[str] = _VENDOR_ALIASES.get(low)
    return alias is not None and alias in vendor


def _resolve_device(backend:str, device_sel:Optional[str])->Tuple[Any, str]:
    """Resolve a device within a backend: None/index/name substring ("name:ordinal" disambiguates); returns (device object, cache key)."""
    devices:List[Any] = _backend_devices(backend)

    if not devices:
        raise RuntimeError(f"no {backend} devices found")

    if device_sel is None or device_sel.strip() == "":
        return devices[0], devices[0].name

    sel:str = device_sel.strip()
    index:int = 0
    if ":" in sel:
        name_part, idx_part = sel.rsplit(":", 1)
        if idx_part.strip().isdigit() and name_part.strip():
            sel = name_part.strip()
            index = int(idx_part)

    if sel.isdigit():
        i:int = int(sel)
        if not (0 <= i < len(devices)):
            raise ValueError(f"device index {i} out of range (0..{len(devices)-1}) for backend '{backend}'")
        return devices[i], devices[i].name

    matches:List[Any] = [d for d in devices if _device_match_keys(sel, d)]
    if not matches:
        raise ValueError(
            f"no {backend} device matches '{device_sel}'; available: " + ", ".join(d.name for d in devices))
    if index >= len(matches):
        raise ValueError(f"device match index {index} out of range for '{sel}' ({len(matches)} matches)")
    return matches[index], matches[index].name


def list_devices()->List[str]:
    """List all available "backend:device" selector strings, ordered by DEFAULT_BACKEND_ORDER.

    Returns a list like ["cuda:0", "hip:0", "opencl:0", "opencl:1", ...]; the numbering
    matches the device order used by plain-numeric selectors; backends whose runtime is
    missing are skipped. Results can be passed directly to `get_device()` or kernel brackets.
    """
    selectors:List[str] = []
    for backend in DEFAULT_BACKEND_ORDER:
        try:
            devices:List[Any] = _backend_devices(backend)
        except Exception:
            continue
        for i in range(len(devices)):
            selectors.append(f"{backend}:{i}")
    return selectors


# The name this function had before; kept so existing call sites keep working.
available_devices = list_devices


def get_device(selector:str)->Any:
    """Return the Device object for a "backend:device" selector string (the backend's own Device class).

    The backend part supports aliases (opencl/ocl/cl, cuda/cu, hip/rocm/amd); the device part is an
    index or a name/vendor/arch substring ("name:ordinal" disambiguates); with only the device part,
    the default backend (probed, cuda -> hip -> opencl) is used.
    """
    backend:Optional[str]
    device_sel:Optional[str]
    backend, device_sel = _parse_selector(selector)
    if backend is None:
        backend = _default_backend()
    return _resolve_device(backend, device_sel)[0]


def _bind_arguments(func:Callable, args:Tuple[Any, ...],
                    kwargs:Dict[str, Any],
                    skip_first:bool=False)->Tuple[Any, ...]:
    """Turn a call into the positional tuple the launch path expects.

    A kernel is launched positionally: the argument order decides the device parameter
    order, and the work size is read from a named argument. Accepting keywords here
    keeps the call site looking like ordinary Python without pushing that bookkeeping
    into the launch path. Omitted parameters with a default are filled in, so the launch
    always sees one value per device parameter.

    ``func`` is the undecorated function, so the signature is the author's own.
    ``skip_first`` drops the leading parameter of a method, whose receiver is supplied
    separately by the caller.

    A bad call must fail the way the same call to the undecorated function fails, so the
    binding runs through CPython itself (see _signature_stub) instead of a re-worded
    message of our own: the wording, the singular/plural forms, the "Did you mean ...?"
    suggestion and even the mention of keyword-only leftovers then follow the running
    interpreter rather than a hand-written table of phrases.

    A method is checked with its receiver counted, because that is how CPython words the
    error for the undecorated method: `Model.run() takes 3 positional arguments but 4 were
    given`. The receiver is a placeholder -- its value never reaches the kernel, only the
    count does.
    """
    full_signature:Optional[inspect.Signature] = _callable_signature(func, skip_first=False)
    signature:Optional[inspect.Signature] = _callable_signature(func, skip_first)
    if signature is None:
        if kwargs:
            raise TypeError(
                "keyword arguments need a retrievable signature; call positionally") from None
        return args

    # Let CPython bind and reject: it raises the native TypeError for an arity or keyword
    # mismatch, attributed to the kernel's own qualified name and trimmed to the caller.
    assert full_signature is not None
    stub:Callable = _signature_stub(full_signature, func)
    bound:inspect.BoundArguments = _bind_and_reject(
        func, signature, stub, args, kwargs, skip_first)
    qualname:str = _qualname_for_errors(func)
    for name in bound.arguments:
        if name == "self":
            continue
        try:
            _check_annotated_argument(func, name, bound.arguments[name], qualname)
        except ArgumentTypeError as error:
            error.__traceback__ = None
            raise

    ordered:List[Any] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in bound.arguments:
            ordered.append(bound.arguments[name])
        else:
            # bind() accepted the call without it, so the parameter has a default.
            ordered.append(parameter.default)
    return tuple(ordered)


def _bound_arguments_for_call(func:Callable, args:Tuple[Any, ...],
                              kwargs:Dict[str, Any],
                              skip_first:bool=False)->Tuple[Any, ...]:
    """``_bind_arguments`` as the entry points call it (see _native_argument_errors)."""
    return _bind_arguments(func, args, kwargs, skip_first)


def _native_argument_errors(entry:Callable[..., Any])->Callable[..., Any]:
    """Give an entry point native-python's single-frame traceback for a bad call.

    Native python blames the call site and nothing else; a decorated call would otherwise
    also show this module's ``__call__`` and the signature stub, so the same mistake looked
    different once a function was decorated. The trimming happens here, at the entry point,
    because clearing the traceback deeper inside and re-raising lets every frame the
    exception travels through push itself back on again.
    """
    @functools.wraps(entry)
    def wrapper(*args:Any, **kwargs:Any)->Any:
        try:
            return entry(*args, **kwargs)
        except (ArgumentTypeError, TypeError) as error:
            error.__traceback__ = None
            raise

    return wrapper


def _callable_signature(func:Callable, skip_first:bool)->Optional[inspect.Signature]:
    """The signature a kernel call is checked against, with a method's receiver removed."""
    try:
        signature:inspect.Signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    if not skip_first:
        return signature
    parameters:List[inspect.Parameter] = list(signature.parameters.values())[1:]
    return signature.replace(parameters=parameters)


_STUB_SCACHE:Dict[str, Callable] = {}


class ArgumentTypeError(TypeError):
    """A TypeError raised by this module's argument checks.

    Marked as its own class so the call boundary can trim the traceback to the caller for
    the errors this module words itself, the same treatment the interpreter's own binding
    errors get. Callers see a plain TypeError, exactly as they would from the undecorated
    function.
    """

# Stands in for a method's receiver when CPython is asked to word an argument error.
# Never reaches the kernel: it exists so the error counts the receiver like native python.
_RECEIVER_PLACEHOLDER:Any = object()


def _signature_stub(signature:inspect.Signature, func:Callable)->Callable:
    """A stand-in function with this signature, used only to raise CPython's own errors.

    Calling it performs the real argument binding, so a wrong call fails with exactly the
    text the undecorated function would have produced -- including the parts a
    reimplementation tends to miss ("but 3 were given", "1 required positional argument",
    "Did you mean 'b'?"). Annotations are dropped because they cannot always be
    stringified, and no parameter type reaches this function: only the shape matters.

    Stubs are cached per signature text and qualified name, so a repeated bad call does
    not recompile the same throwaway function.
    """
    source_signature:str = str(signature)
    qualname:str = _qualname_for_errors(func)
    cache_key:str = qualname + source_signature
    stub:Optional[Callable] = _STUB_SCACHE.get(cache_key)
    if stub is not None:
        return stub

    body:str = ("from __future__ import annotations\n"
                f"def _signature_stub{source_signature}:\n"
                f"    return None\n")
    namespace:Dict[str, Any] = {"__name__": "pygpgpu.kernel._signature_stub"}
    exec(compile(body, "<kernel signature>", "exec"), namespace)  # noqa: S102 - a builder, not a container
    stub = namespace["_signature_stub"]
    # CPython prints the qualified name in its message, so an error reads
    # `Model.method() ...` exactly as it would for the undecorated method.
    stub.__qualname__ = qualname
    stub.__name__ = qualname
    _STUB_SCACHE[cache_key] = stub
    return stub


def _qualname_for_errors(func:Callable)->str:
    """The name CPython would print for this callable in an argument error."""
    qualname:str = getattr(func, "__qualname__", None) or getattr(func, "__name__", "kernel")
    return qualname


# What a kernel parameter's annotation says the argument should be. Only the two kinds a
# kernel can carry are recognised; an annotation that is absent or means something else
# leaves the argument unchecked, so this never rejects a call the launch path accepts.
_ARRAY_ANNOTATIONS:Set[str] = {"np.ndarray", "ndarray", "numpy.ndarray"}
_SCALAR_ANNOTATIONS:Set[str] = {"int", "float", "bool", "np.int32", "np.int64", "np.float32",
                                "np.float64", "np.bool_", "np.uint8", "np.uint32", "np.uint64"}


def _annotation_kind(func:Callable, name:str)->str:
    """'array', 'scalar' or '' for one parameter's annotation.

    An annotation may arrive as a type object or (under ``from __future__ import
    annotations``) as its source text, so both forms are recognised. Anything else means
    "no opinion": a kernel is specialized per argument type at call time (see
    infer_ctype), which is why the annotation is documentation rather than a constraint
    the launch has to honour. Name matching is by text because resolving a forward
    reference would import a module just to reject a call.
    """
    annotations:Any = getattr(func, "__annotations__", None) or {}
    annotation:Any = annotations.get(name, None)
    if annotation is None:
        return ""
    if isinstance(annotation, type):
        if annotation is np.ndarray:
            return "array"
        return "scalar" if annotation in (bool, int, float) else ""
    if isinstance(annotation, str):
        text:str = annotation.strip().strip("'\"")
        if text in _ARRAY_ANNOTATIONS:
            return "array"
        if text in _SCALAR_ANNOTATIONS:
            return "scalar"
        return ""
    return ""


def _check_annotated_argument(func:Callable, name:str, value:Any, qualname:str) -> None:
    """Reject an argument whose type contradicts the parameter's annotation.

    This is the argument check a reader expects from the signature they wrote: passing a
    scalar where an array is annotated used to surface much later as a confusing device
    error about the kernel body, which pointed at the kernel source instead of the call.
    """
    kind:str = _annotation_kind(func, name)
    if not kind:
        return
    if kind == "array" and not isinstance(value, np.ndarray):
        raise ArgumentTypeError(
            f"{qualname}() argument '{name}' must be a numpy array, not "
            f"{type(value).__name__}")
    if kind == "scalar" and isinstance(value, np.ndarray):
        raise ArgumentTypeError(
            f"{qualname}() argument '{name}' must be a scalar, not "
            f"{type(value).__name__}")


def _bind_and_reject(func:Callable, signature:inspect.Signature,
                     stub:Callable, args:Tuple[Any, ...],
                     kwargs:Dict[str, Any], skip_first:bool)->inspect.BoundArguments:
    """Let CPython bind the call, re-raising its TypeError from the caller's own frame.

    Native python reports a bad call with one frame -- the call site -- and the traceback
    here would otherwise also show this module's ``__call__`` and the stub's call, which is
    what makes the same mistake look different once a function is decorated. Clearing the
    traceback and re-raising at this boundary leaves exactly the caller's frame, the same
    shape ``_translate_errors`` gives a translation failure.
    """
    try:
        if skip_first:
            stub(_RECEIVER_PLACEHOLDER, *args, **kwargs)
        else:
            stub(*args, **kwargs)
        return signature.bind(*args, **kwargs)
    except TypeError as error:
        error.__traceback__ = None
        raise


def _caller_traceback(traceback:Any)->Any:
    """The first frame of ``traceback`` that is not inside this package."""
    if traceback is None:
        return None
    package_dir:str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prefix:str = package_dir + os.sep
    cursor:Any = traceback
    while cursor is not None:
        filename:str = cursor.tb_frame.f_code.co_filename
        # `-c`, stdin and exec'd code use a pseudo-filename that abspath() would resolve
        # against the working directory, which can itself sit under the package.
        pseudo:bool = filename.startswith("<") and filename.endswith(">")
        if pseudo or not os.path.abspath(filename).startswith(prefix):
            return cursor
        cursor = cursor.tb_next
    return None


def _user_frame_syntax_error(error:SyntaxError)->SyntaxError:
    """Drop the library frames from a SyntaxError raised while translating a kernel.

    Kept for callers that want a trimmed chain rather than an empty one; the launch path
    itself clears the traceback (see _translate_errors), because a re-raise always prepends
    the frame doing the re-raising.
    """
    traceback = error.__traceback__
    if traceback is None:
        return error

    package_root:str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_prefix:str = package_root + os.sep

    keep:Any = None
    cursor:Any = traceback
    while cursor is not None:
        filename:str = cursor.tb_frame.f_code.co_filename
        # `-c`, stdin and exec'd code report a pseudo-filename that abspath() would resolve
        # against the working directory -- which can sit under the package; those frames are
        # the caller's, not the library's.
        pseudo:bool = filename.startswith("<") and filename.endswith(">")
        if pseudo or not os.path.abspath(filename).startswith(package_prefix):
            keep = cursor
            break
        cursor = cursor.tb_next
    if keep is None:
        return error
    error.__traceback__ = keep
    return error


class _KernelCall:
    """Bracket-configuration form: k["cuda"][(64, 8)](args) / k[(64, 8)]["cuda"](args), any order."""

    def __init__(self, kernel:"PyKernel", instance:Any,
                 backend:Optional[str]=None, device:Optional[str]=None,
                 global_size:Optional[Tuple[int, ...]]=None, local_size:Optional[Tuple[int, ...]]=None):
        self._kernel:"PyKernel" = kernel
        self._instance:Any = instance
        self._backend:Optional[str] = backend
        self._device:Optional[str] = device
        self._global_size:Optional[Tuple[int, ...]] = global_size
        self._local_size:Optional[Tuple[int, ...]] = local_size

    def __getitem__(self, item:Any)->"_KernelCall":
        if isinstance(item, str):
            backend, device = _parse_selector(item)
            return _KernelCall(
                self._kernel, self._instance,
                (backend if backend is not None else self._backend),
                (device if device is not None else self._device),
                self._global_size, self._local_size)
        global_size, local_size = _parse_work_item(item)
        return _KernelCall(
            self._kernel, self._instance, self._backend, self._device,
            (global_size if global_size is not None else self._global_size),
            (local_size if local_size is not None else self._local_size))

    @_native_argument_errors
    def __call__(self, *args:Any, **kwargs:Any)->Any:
        instance:Any = self._instance
        instance_arg:Optional[Any] = None
        if self._kernel._is_method and instance is None:
            # class form: Klass.method["cuda"](instance, args...)
            if not args:
                raise TypeError(
                    f"{self._kernel.__name__}() is an instance method; pass the instance "
                    f"as the first argument")
            instance_arg = args[0]
            args = args[1:]
            instance = instance_arg
        call_args:Tuple[Any, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=self._kernel._is_method)
        return self._kernel._invoke(
            instance, call_args, self._backend, self._device, self._global_size, self._local_size)

    def _launch(self, launch:str, args:Tuple[Any, ...],
                kwargs:Optional[Dict[str, Any]]=None)->Any:
        """Shared body of the three launch styles, honouring the bracket selectors."""
        instance:Any = self._instance
        if self._kernel._is_method and instance is None:
            if not args:
                raise TypeError(
                    f"{self._kernel.__name__}() is an instance method; pass the instance "
                    f"as the first argument")
            instance = args[0]
            args = args[1:]
        call_args:Tuple[Any, ...] = _bind_arguments(self._kernel.raw_func, args,
                                                    kwargs or {},
                                                    skip_first=self._kernel._is_method)
        return self._kernel._invoke(
            instance, call_args, self._backend, self._device,
            self._global_size, self._local_size, launch=launch)

    @_native_argument_errors
    def submit(self, *args:Any, **kwargs:Any)->concurrent.futures.Future:
        """Non-blocking launch of the bracketed form, e.g. ``k["cuda"].submit(...)``."""
        return self._launch("submit", args, kwargs)

    @_native_argument_errors
    def async_call(self, *args:Any, **kwargs:Any)->asyncio.Future:
        """Non-blocking asyncio launch of the bracketed form, e.g. ``k["cuda"].async_call(...)``."""
        return self._launch("async", args, kwargs)


class _BoundKernelMethod:
    """Bound instance-method form: instance.kernel_method(...)"""

    def __init__(self, kernel:"PyKernel", instance:Any):
        self._kernel:"PyKernel" = kernel
        self._instance:Any = instance

    def __getitem__(self, item:Any)->_KernelCall:
        return _KernelCall(self._kernel, self._instance)[item]

    @_native_argument_errors
    def __call__(self, *args:Any, **kwargs:Any)->Any:
        call_args:Tuple[Any, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=True)
        return self._kernel._invoke(self._instance, call_args, None, None, None, None)

    @_native_argument_errors
    def submit(self, *args:Any, **kwargs:Any)->concurrent.futures.Future:
        """Non-blocking launch: ``instance.kernel_method.submit(...)``."""
        call_args:Tuple[Any, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=True)
        return self._kernel._invoke(self._instance, call_args, None, None, None, None,
                                    launch="submit")

    @_native_argument_errors
    def async_call(self, *args:Any, **kwargs:Any)->asyncio.Future:
        """Non-blocking asyncio launch: ``instance.kernel_method.async_call(...)``."""
        call_args:Tuple[Any, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=True)
        return self._kernel._invoke(self._instance, call_args, None, None, None, None,
                                    launch="async")


class PyKernel:
    """Implementation of the `@kernel` decorator: translates the python function into backend source at call time, then compiles and runs it.

    Decorating class methods is supported: self is dropped during translation, self.xxx attribute
    references in the method body become extra parameters _self_xxx, filled from the instance at call
    time; staticmethod is supported too (either decoration order). A method may also call other
    methods of its own class (self.helper(...)), a static method of any class, or a module function;
    the attributes those helpers read become kernel arguments as well.

    Backend/device/work size are given with brackets at call time (same style as handwritten kernels);
    0, 1 or 2 brackets may be used, in any order::

        k(a, out)                       # default backend (probed, cuda -> hip -> opencl)/device, work size automatic
        k["cuda"](a, out)               # pick the backend
        k["opencl:gfx1036"](a, out)     # backend + device (index or name substring, "name:ordinal" disambiguates)
        k[(64, 8)](a, out)              # global/local (None means automatic)
        k["cuda"][(64, 8)](a, out)      # combined, any order
    """

    def __init__(self, func:Callable, backend:str, explicit_backend:bool=False):
        if isinstance(func, classmethod):
            raise SyntaxError("@kernel cannot decorate classmethods (cls semantics are host-only)") from None
        if isinstance(func, staticmethod):
            func = func.__func__

        self.raw_func:Callable = func
        self.backend:str = backend
        # True when the decorator named a backend, so set_default_device() must not
        # override it. A backend chosen by probing is only a snapshot, not a binding.
        self._explicit_backend:bool = explicit_backend
        self.is_pykernel:bool = True
        self.__name__:str = func.__name__
        self.__doc__:str = func.__doc__ or ""

        arg_names:List[str] = list(func.__code__.co_varnames[:func.__code__.co_argcount])
        self._is_method:bool = bool(arg_names and arg_names[0] == "self")
        self._self_param:str = (arg_names[0] if self._is_method else "")
        # A decorated body may read self.<attr> or call self.<helper>(...), and only the
        # latter can tell a method from a value -- which needs the instance's class, known
        # at call time rather than now. The attribute set is therefore collected in
        # _invoke; the decoration-time body scan this used to cache is gone.

        self._cache:Dict[str, Tuple[Any, List[Tuple[int, int]], List[Tuple[int, int]], List[int]]] = {}
        # The python backend executes the body as-is, so there is nothing to specialize or
        # cache: one runner per kernel object serves every call.
        self._python_runner:Any = None
        self._source_sig:str = self._source_signature()

    def __get__(self, obj:Any, objtype:Optional[type]=None)->Any:
        # descriptor protocol: instance access returns the bound form, class access/static methods return self
        if obj is None or not self._is_method:
            return self
        return _BoundKernelMethod(self, obj)

    def __getitem__(self, item:Any)->_KernelCall:
        return _KernelCall(self, None)[item]

    @_native_argument_errors
    def __call__(self, *args:Any, **kwargs:Any)->Any:
        if self._is_method:
            if not args:
                raise TypeError(
                    f"{self.__name__}() is an instance method; call it on an instance "
                    f"or pass the instance as the first argument")
            call_args:Tuple[Any, ...] = _bind_arguments(self.raw_func, args[1:], kwargs, skip_first=True)
            return self._invoke(args[0], call_args, None, None, None, None)
        call_args = _bind_arguments(self.raw_func, args, kwargs)
        return self._invoke(None, call_args, None, None, None, None)

    @_native_argument_errors
    def submit(self, *args:Any, **kwargs:Any)->concurrent.futures.Future:
        """Launch without waiting, returning a ``concurrent.futures.Future``.

        Same arguments and same backend/work-size selection as calling the kernel
        directly; the difference is that this returns as soon as the launch is
        enqueued instead of blocking until the kernel finishes and its results
        are read back. The future completes when every transfer has landed, so
        the output arrays are only safe to read from its result (or after
        ``.result()``/``.wait()``).
        """
        if self._is_method:
            if not args:
                raise TypeError(
                    f"{self.__name__}() is an instance method; call it on an instance "
                    f"or pass the instance as the first argument")
            call_args:Tuple[Any, ...] = _bind_arguments(self.raw_func, args[1:], kwargs, skip_first=True)
            return self._invoke(args[0], call_args, None, None, None, None, launch="submit")
        call_args = _bind_arguments(self.raw_func, args, kwargs)
        return self._invoke(None, call_args, None, None, None, None, launch="submit")

    @_native_argument_errors
    def async_call(self, *args:Any, **kwargs:Any)->asyncio.Future:
        """Launch without waiting, returning an ``asyncio.Future``.

        The asyncio counterpart of ``submit``; use ``await`` on the result. The
        kernel is enqueued immediately, so awaiting only waits for completion,
        and the returned future resolves once the read-back has finished.
        """
        if self._is_method:
            if not args:
                raise TypeError(
                    f"{self.__name__}() is an instance method; call it on an instance "
                    f"or pass the instance as the first argument")
            call_args:Tuple[Any, ...] = _bind_arguments(self.raw_func, args[1:], kwargs, skip_first=True)
            return self._invoke(args[0], call_args, None, None, None, None, launch="async")
        call_args = _bind_arguments(self.raw_func, args, kwargs)
        return self._invoke(None, call_args, None, None, None, None, launch="async")

    def _invoke(self, instance:Any, args:Tuple[Any, ...],
                backend_sel:Optional[str], device_sel:Optional[str],
                global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                launch:str = "call")->Any:
        try:
            return self._invoke_inner(instance, args, backend_sel, device_sel,
                                      global_size, local_size, launch)
        except SyntaxError as caught:
            # Trim to the caller's frame and re-raise with that traceback. CPython prepends one
            # frame for the re-raise itself, which is why the exception carries the caller's
            # line rather than this one: the syntax rendering uses the fields on the
            # exception, not the chain.
            caller = _caller_traceback(caught.__traceback__)
            raise caught.with_traceback(caller) from None

    def _invoke_inner(self, instance:Any, args:Tuple[Any, ...],
                      backend_sel:Optional[str], device_sel:Optional[str],
                      global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                      launch:str = "call")->Any:
        if backend_sel is not None:
            backend:str = backend_sel
        elif getattr(self, "_explicit_backend", False):
            # Decorated as @kernel(backend=...): that backend wins over any default.
            backend = self.backend
        else:
            # No bracket and no explicit backend: consult set_default_device(), read now
            # rather than at decoration time so a later call takes effect.
            backend, default_device = _default_target()
            if device_sel is None:
                device_sel = default_device
        # Record what this call actually used, so `k.backend` describes the last launch
        # rather than the snapshot taken when the decorator ran.
        self.backend = backend
        if backend == "python":
            return self._invoke_python(instance, args, device_sel, global_size, local_size, launch)
        device_obj:Any
        device_key:str
        device_obj, device_key = _resolve_device(backend, device_sel)
        device_max_group, device_max_items = device_limits(device_obj)

        self_values:List[Any] = []
        self_attrs:List[str] = []
        owner:Optional[type] = None
        if self._is_method:
            if instance is None:
                raise TypeError(f"{self.__name__}() requires an instance")
            # The decorated body was scanned at decoration time, before the class
            # existed, so self.<helper> could only look like data. Scan again now
            # that the instance is here: calling a self-method makes the attributes
            # *it* reads kernel arguments too, recursively.
            owner = type(instance)
            self_attrs = scan_self_attrs(self.raw_func, self._self_param, owner)
            for name in self_attrs:
                if not hasattr(instance, name):
                    raise SyntaxError(
                        f"instance of {owner.__name__} has no attribute '{name}' "
                        f"required by kernel {self.__name__}") from None
                value:Any = getattr(instance, name)
                if callable(value) and not isinstance(value, np.ndarray):
                    raise SyntaxError(
                        f"self.{name} is a function, not a value: a function cannot be passed "
                        f"to a device (call it as self.{name}(...) instead)") from None
                self_values.append(value)

        # An arbitrary python object becomes a synthesized structure before its type is
        # inferred: the structure *is* the argument type, and every step after this sees the
        # same `ctypes.Structure` a caller would have declared by hand. The binding records
        # the pairing, so member writes travel back to the caller's object after the launch.
        # The receiver travels along too: a body that builds its own value (`ray = Ray(...)`)
        # types the construction's arguments from what the receiver's members are, and the
        # receiver's instance is the only place that answer lives.
        object_binding:ObjectBinding = ObjectBinding()
        args = convert_object_arguments(self.raw_func, args, self._resolve_global,
                                        self._self_param, object_binding,
                                        self_value=instance)
        arg_types:List[Any] = [infer_ctype(a) for a in args]
        self_types:List[Any] = []
        for name, value in zip(self_attrs, self_values):
            try:
                self_types.append(infer_ctype(value))
            except SyntaxError as e:
                raise SyntaxError(f"self.{name}: {e}") from None

        sig:str = type_sig(arg_types + self_types)
        cache_key:str = f"{backend}|{device_key}|{sig}"
        entry = self._cache.get(cache_key)
        if entry is None:
            # Create the device objects for any Image before compiling. On the driver
            # this was developed against, loading a module that contains texture or
            # surface calls makes every later texture/surface creation fail, so an
            # image bound after compilation would never get its device storage.
            self._bind_images(full_values=args, backend=backend)
            entry = self._compile(args, arg_types, self_types, self_attrs, owner, sig,
                                  backend, device_obj, instance)
            self._cache[cache_key] = entry
        (kernel_obj, shape_sources, work_size_dims, local_sizes, uses_global_id,
         image_samplers, returns, outer_vars, launch_meta) = entry
        # One entry per magic loop, launched in order; a single-loop body has exactly one.
        active_kernels:List[Any] = list(launch_meta.get("kernels") or [kernel_obj])
        # The extent arguments each phase kernel declares, in phase order.
        kernel_shapes:List[List[Tuple[int, int]]] = list(
            launch_meta.get("kernel_shapes") or [[] for _ in active_kernels])
        kernel_created:List[List[str]] = list(
            launch_meta.get("kernel_created") or [[] for _ in active_kernels])

        full:List[Any] = [_kernel_scalar_argument(value, t)
                          for value, t in zip(list(args) + list(self_values),
                                              arg_types + self_types)]
        own_count:int = len(full)
        # Outer-scope values are read before the launch arguments are built: a shape
        # source can name an outer array, and `full` is indexed by the device parameter
        # order (see _build_launch_args), so the outer values have to be part of it.
        outer_state:Dict[str, Any] = self._bind_outer_values(outer_vars, backend)
        # Objects the kernel body creates are built here, on every call, exactly as the
        # interpreter would at that point in the body. They travel as trailing parameters.
        # A method's creation may read `self` (`np.zeros((self.height, ...))`), so the
        # instance is bound too.
        created:Dict[str, Any] = _host_created_values(
            self.raw_func, tuple(args), instance)[0]
        runtime_values:List[Any] = list(outer_state["full"]) + _created_arguments(created)
        return_state:Dict[str, Any] = (self._alloc_return_slots(returns)
                                      if returns else {})
        # Launch order must match the device signature: own arguments, shape extents, then
        # the values the host supplies (outer scope, then created objects). The shape
        # extents are read from `full` by device-parameter index, so the runtime values are
        # part of `full` but are appended to the launch afterwards.
        full.extend(runtime_values)
        # `full` indexes like the device parameter list, which is what shape_sources were
        # recorded against. Validate the dimensions first: an index past an array's rank
        # used to raise a bare `IndexError: tuple index out of range`, which says nothing
        # about which array or loop was wrong.
        _check_shape_indices(self.__name__, full, shape_sources, work_size_dims, launch_meta)

        def build_phase_args(extents:List[Tuple[int, int]],
                             extra_created:Optional[List[str]])->List[Any]:
            """The launch arguments for one phase kernel.

            The shared prefix is the device parameter list up to `own_count`; the extents
            follow it because the translator appends them to each kernel's signature. The
            created objects differ per phase -- `mid = np.zeros(4)` between two loops is a
            parameter of the pass that reads it and of no other -- so `extra_created` names
            the ones this kernel expects.
            """
            phase_args:List[Any] = list(full[:own_count])
            for param_index, dim in extents:
                phase_args.append(int(full[param_index].shape[dim]))
            # The signature declares the sampler and local-memory arguments right after the
            # shape extents, and the outer/created values last (see render_kernel), so they
            # are appended in that order here. Getting this wrong shifts every later
            # argument: the local pointer would land where an outer scalar is expected.
            if backend == "opencl":
                for param_index in image_samplers:
                    phase_args.append(_image_sampler(full[param_index]))
                for local_bytes in local_sizes:
                    phase_args.append(local_bytes)
            phase_args.extend(outer_state["full"])
            for name, value in created.items():
                if extra_created is None or name in extra_created:
                    phase_args.append(_kernel_scalar_argument(value, infer_ctype(value)))
            if return_state:
                phase_args.extend(return_state["buffers"])
                phase_args.append(return_state["claim"])
                phase_args.append(return_state["site"])
            return phase_args

        launch_args:List[Any] = build_phase_args(shape_sources, None)
        # Vector values travel in the layout the backend's ABI defines: OpenCL pads a
        # 3-component vector to four slots, CUDA/HIP take the host's three. The host keeps
        # the three-component form everywhere, so each argument is converted here and its
        # result copied back once the launch has finished.
        unpackers:List[Any] = []
        for index, argument in enumerate(launch_args):
            packed_value, unpack = pack(argument, backend)
            if packed_value is not argument:
                launch_args[index] = packed_value
            if unpack is not None:
                unpackers.append((packed_value, unpack))
        if global_size is None and work_size_dims:
            # A dim of -1 is an explicit gpu.work_size(x) declaration: that array contributes
            # all of its dimensions, matching what np.ndindex(x.shape) would give. An entry
            # with no parameter index is a literal count, as in np.ndindex(2, 3).
            extents_list:List[int] = []
            for pi, dim in work_size_dims:
                if pi is None:
                    extents_list.append(dim)
                    continue
                if dim == -2:
                    # A host-side integer supplied as an argument, used as an extent:
                    # `np.ndindex(self.image_height, self.image_width)`.
                    extents_list.append(int(full[pi]))
                    continue
                shape:Tuple[int, ...] = tuple(int(s) for s in full[pi].shape)
                if dim >= 0:
                    extents_list.append(shape[dim])
                else:
                    extents_list.extend(shape)
            if len(extents_list) > 3:
                source:str = ""
                if work_size_dims[0][0] is not None:
                    source = (f" from an array of shape "
                              f"{tuple(int(s) for s in full[work_size_dims[0][0]].shape)}")
                raise ValueError(
                    f"the magic loop of {self.__name__} declared {len(extents_list)} "
                    f"dimensions{source}; at most 3 are supported")
            # `np.ndindex(arr.shape)` means the array's whole shape, but the translator only
            # sees a pointer and cannot know its rank, so it takes one dimension per loop
            # name. `for j, i in np.ndindex(image.shape)` over an RGB image would therefore
            # iterate two of the three axes and never vary the channel. Check it here, where
            # the rank is known.
            whole:int = int(launch_meta.get("whole_shape_index", -1))
            loop_names:int = int(launch_meta.get("loop_names", 0))
            if whole >= 0 and loop_names:
                rank:int = len(tuple(int(s) for s in full[whole].shape))
                if rank != loop_names:
                    raise ValueError(
                        f"{self.__name__}(): the magic loop unpacks {loop_names} index "
                        f"name{'s' if loop_names != 1 else ''} but np.ndindex(...) iterates "
                        f"the whole shape of '{launch_meta['param_names'].get(whole, 'the array')}'"
                        f", which has {rank} dimension{'s' if rank != 1 else ''} "
                        f"{tuple(int(s) for s in full[whole].shape)}. Give one name per "
                        f"dimension, or iterate an explicit subset, e.g. "
                        f"np.ndindex(image.shape[0], image.shape[1])")
            # The loop has to name one index per dimension of its domain: `for j, i in
            # np.ndindex(image.shape)` over an RGB image would silently iterate only two of
            # the three dimensions, so the third axis never varies. Say so instead.
            if loop_names and loop_names != len(extents_list):
                source = ""
                if work_size_dims[0][0] is not None:
                    source = (f" (the array has shape "
                              f"{tuple(int(s) for s in full[work_size_dims[0][0]].shape)})")
                raise ValueError(
                    f"{self.__name__}(): the magic loop unpacks {loop_names} index "
                    f"name{'s' if loop_names != 1 else ''} but its iteration domain has "
                    f"{len(extents_list)} dimension{'s' if len(extents_list) != 1 else ''}"
                    f"{source}. Give one name per dimension, or iterate an explicit subset, "
                    f"e.g. np.ndindex(image.shape[0], image.shape[1])")
            extents:Tuple[int, ...] = tuple(extents_list)
            if local_size is None:
                global_size, local_size = detect_work_size(
                    256, extents, max_work_group_size=device_max_group,
                    max_work_item_sizes=device_max_items, allow_padding=True)
            elif len(local_size) == len(extents):
                # A user-supplied local size wins; round each extent up to a multiple of it.
                global_size = tuple(-(-extent // l) * l for extent, l in zip(extents, local_size))
            else:
                global_size = extents
        if global_size is None and local_size is not None:
            raise ValueError(
                "local work size requires a global size: specify both in the bracket "
                "k[(global, local)] or rely on a magic loop to derive the global size")
        if global_size is None and uses_global_id:
            # A gid-style kernel has no loop to read the domain from. Falling back to the
            # backend's buffer-shape guess used to silently launch a tiny grid (e.g. one work
            # item for a 100000-element reduction) and produce a wrong result, so refuse.
            raise ValueError(
                f"{self.__name__}() dispatches on gpu.get_global_id but declares no iteration "
                f"domain, so the work size cannot be derived. Either give it a magic loop "
                f"(`for i in range(x.shape[0]): ...`), declare `gpu.work_size(x)` in the body, "
                f"or pass a bracket: {self.__name__}[(total, local)](...)")
        if global_size is not None:
            # Uniform total-work-item semantics (OpenCL/SYCL): global = total number of work items, must be divisible by local;
            # cuda's gridDim is a block count, derived from total/local, transparent to the user
            if local_size is None:
                local_size = local_size_for_global(
                    global_size, max_work_group_size=device_max_group,
                    max_work_item_sizes=device_max_items)
            if len(local_size) != len(global_size):
                raise ValueError(
                    f"global {global_size} and local {local_size} work sizes must have the same dimension")
            for g, l in zip(global_size, local_size):
                if g % l != 0:
                    raise ValueError(
                        f"global work size {global_size} must be divisible by local {local_size}; "
                        f"use None for an automatically derived local size")
            geometry = (tuple(g // l for g, l in zip(global_size, local_size))
                        if backend in ("cuda", "hip") else global_size)
            phase_geometry = (geometry, local_size)
        else:
            phase_geometry = None
        # Every phase of the body gets the same geometry: they are passes over one domain.
        # Each is launched in turn on the stream this call owns, so the arrays that travel
        # between them stay on the device instead of going back to the host and returning.
        if launch == "submit":
            future = active_kernels[0].submit(*launch_args)
            if return_state or outer_state["writable"]:
                return self._with_returns(future, return_state, returns, full, outer_state, created)
            return future
        if launch == "async":
            future = active_kernels[0].async_call(*launch_args)
            if return_state or outer_state["writable"]:
                return self._with_returns(future, return_state, returns, full, outer_state, created)
            return future
        for position, phase_kernel in enumerate(active_kernels):
            # A phase kernel declares only the shapes *it* indexes, so its argument list is
            # rebuilt from the same sources with its own extents -- a later phase commonly
            # indexes an array the first one never named.
            phase_shapes:List[Tuple[int, int]] = (
                kernel_shapes[position] if position < len(kernel_shapes) else [])
            bound_kernel = (phase_kernel[phase_geometry] if phase_geometry is not None
                            else phase_kernel)
            phase_created_names:List[str] = (
                kernel_created[position] if position < len(kernel_created) else [])
            # A phase is described by its extents *and* by the objects it created: a pass
            # that reads an array built between the loops declares it, one that does not
            # must not receive it.
            if (phase_shapes == shape_sources
                    and phase_created_names == list(created)):
                bound_kernel(*launch_args)
            else:
                bound_kernel(*build_phase_args(phase_shapes, phase_created_names))
        for packed_value, unpack in unpackers:
            unpack(packed_value)
        result:Any = (self._finish_returns(return_state, returns, full, created)
                      if return_state else None)
        self._write_back_outer(outer_state)
        if object_binding.written:
            object_binding.write_back()
        return result

    # ---------- outer-scope variables ----------

    def _bind_outer_values(self, outer_vars:Dict[str, Any], backend:str)->Dict[str, Any]:
        """The launch arguments carrying this call's outer-scope values.

        A read-only outer variable is passed by value, refreshed on every call, so a
        changed scope value is seen immediately -- that is the whole point of
        parameterizing it instead of inlining it. A declared `nonlocal`/`global` scalar
        travels as a one-element buffer, because the device writes through the pointer;
        the buffer is copied back into the scope once the launch has completed (see
        _write_back_outer and _with_returns). An outer array is passed as itself, and the
        backend's read/write analysis reads it back when the kernel wrote it.
        """
        args:List[Any] = []
        full:List[Any] = []
        writable:List[Dict[str, Any]] = []
        for name, outer in outer_vars.items():
            kind:str = outer["kind"]
            value:Any = self._read_outer(name, kind, outer.get("scope"))
            if is_ptr(outer["type"]):
                if value is None:
                    raise ValueError(
                        f"outer array '{name}' is undefined at launch time")
                args.append(value)
                full.append(value)
                continue
            if not outer["writable"]:
                converted:Any = _kernel_scalar_argument(value, outer["type"])
                args.append(converted)
                full.append(converted)
                continue
            buffer:np.ndarray = np.zeros(1, dtype=_outer_slot_dtype(outer["type"]))
            buffer[0] = _scalar_slot_value(value, outer["type"])
            args.append(buffer)
            full.append(buffer)
            writable.append({"name": name, "kind": kind, "buffer": buffer,
                             "type": outer["type"]})
        return {"args": args, "writable": writable, "full": full}

    def _read_outer(self, name:str, kind:str,
                    scope:Optional[Dict[str, Any]] = None)->Any:
        """The current value of an outer name, read the way python would resolve it.

        ``scope`` is the namespace the translator resolved the name in, which is the
        kernel's module for a name the body reads itself and the *helper's* module for one
        a device helper reads. Both are read live so a changed value is seen on the next
        call, which is why the namespace is carried rather than the value.
        """
        if kind == "closure":
            cells:Dict[str, Any] = {}
            for cell_name, cell in zip(self.raw_func.__code__.co_freevars,
                                       self.raw_func.__closure__ or ()):
                cells[cell_name] = cell
            cell = cells.get(name)
            if cell is None:
                raise ValueError(f"outer variable '{name}' is not in the kernel's closure")
            return cell.cell_contents
        if scope is not None:
            return scope.get(name, None)
        return self.raw_func.__globals__.get(name, None)

    def _write_back_outer(self, outer_state:Dict[str, Any])->None:
        """Copy a declared outer scalar's final value back into its scope."""
        for entry in outer_state.get("writable", ()):
            value:Any = _outer_scope_value(entry["buffer"][0], entry["type"])
            self._store_outer(entry["name"], entry["kind"], value)

    def _store_outer(self, name:str, kind:str, value:Any)->None:
        """Assign an outer name, through its closure cell or the module globals."""
        if kind == "closure":
            for cell_name, cell in zip(self.raw_func.__code__.co_freevars,
                                       self.raw_func.__closure__ or ()):
                if cell_name == name:
                    cell.cell_contents = value
                    return
            raise ValueError(f"outer variable '{name}' is not in the kernel's closure")
        self.raw_func.__globals__[name] = value

    def _alloc_return_slots(self, returns:Dict[str, Any])->Dict[str, Any]:
        """One device buffer per value slot, plus the shared claim and site words.

        Every buffer is zero-initialised: the claim word then starts unclaimed, so a
        launch where no work item returned reads back as None rather than as garbage.
        """
        buffers:List[np.ndarray] = [np.zeros(1, dtype=dtype) for dtype in returns["slot_types"]]
        return {"buffers": buffers,
                "claim": np.zeros(1, dtype=np.int32),
                "site": np.zeros(1, dtype=np.int32)}

    def _finish_returns(self, return_state:Dict[str, Any],
                        returns:Dict[str, Any], full:List[Any],
                        created:Optional[Dict[str, Any]]=None)->Any:
        """Rebuild the python result of a finished launch.

        The winning return site decides the shape: `return a, b` gives a tuple,
        `return value` gives the bare value, and a bare `return` gives None. A launch in
        which no work item reached any `return` gives None too, exactly as a python
        function does when no returning branch was taken.

        An array position resolves to the host's own object: a parameter comes from the
        launch arguments, and an object the body created comes from `created`. Either way
        the caller receives the same object, never a copy.
        """
        if not int(return_state["claim"][0]):
            return None
        site:int = int(return_state["site"][0]) - 1
        layout:Dict[str, Any] = returns["sites"][site]
        if not layout["positions"]:
            # The winning site was a bare `return` (or `return None`).
            return None
        values:List[Any] = []
        for entry in layout["positions"]:
            if entry["kind"] == "array":
                name:str = entry["name"]
                if created is not None and name in created:
                    values.append(created[name])
                else:
                    values.append(full[returns["arg_indices"][name]])
            else:
                dtype:Any = returns["slot_types"][entry["slot"]]
                kind:str = returns["slot_kinds"][entry["slot"]]
                values.append(_return_value(return_state["buffers"][entry["slot"]][0], dtype, kind))
        return tuple(values) if layout["is_tuple"] else values[0]

    def _with_returns(self, future:Any,
                      return_state:Dict[str, Any],
                      returns:Optional[Dict[str, Any]],
                      full:List[Any],
                      outer_state:Optional[Dict[str, Any]]=None,
                      created:Optional[Dict[str, Any]]=None)->Any:
        """Wrap an asynchronous launch so its future resolves to the returned value.

        The backend future resolves once every read-back has completed (see the kernel
        classes), so by the time this callback runs the slots hold the winner's values.
        A backend future may be bound to a loop other than the caller's (the python
        backend installs one when none is running), so an asyncio result is delivered
        through a fresh future on the caller's running loop when there is one, and is
        otherwise completed in place on whatever loop the backend future uses.
        """
        is_asyncio:bool = isinstance(future, asyncio.Future)
        deliver:Any = concurrent.futures.Future()
        try:
            caller_loop:Any = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None
        if is_asyncio:
            deliver = caller_loop.create_future() if caller_loop is not None else future

        def settle(*_args:Any)->None:
            error:Optional[BaseException] = None
            try:
                error = future.exception()
            except BaseException as caught:  # cancelled, or never resolved
                error = caught

            def apply() -> None:
                if deliver.done():
                    return
                if error is not None:
                    deliver.set_exception(error)
                else:
                    # Write-back happens here, after the launch has completed and its
                    # buffers have been read back, so a declared outer scalar holds the
                    # value the kernel left in it by the time the caller sees the result.
                    if outer_state is not None:
                        self._write_back_outer(outer_state)
                    deliver.set_result(self._finish_returns(return_state, returns, full, created)
                                       if return_state else None)

            if is_asyncio:
                try:
                    future.get_loop().call_soon_threadsafe(apply)
                except RuntimeError:
                    pass  # the loop was closed before the launch finished
                return
            apply()

        if future.done():
            settle()
        else:
            future.add_done_callback(settle)
        return deliver

    def _invoke_python(self, instance:Any, args:Tuple[Any, ...],
                       device_sel:Optional[str],
                       global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                       launch:str)->Any:
        """Run the body in the Python interpreter (no translation, no device).

        The backend is device-free, so a device selector in the bracket is rejected
        rather than ignored; the work size and launch style mean what they mean on the
        GPU backends. The runner decides what a work item is (see python_backend.py): a
        magic-loop kernel runs once as written, a gid-style kernel runs once per work
        item of its domain.
        """
        if device_sel is not None:
            raise ValueError(
                f"the python backend has no devices, so '{self.__name__}[\"python:{device_sel}\"]' "
                f"has no meaning; use {self.__name__}[\"python\"]")
        if self._is_method and instance is None:
            raise TypeError(f"{self.__name__}() requires an instance")
        runner:Any = self._python_runner
        if runner is None:
            runner = PythonKernel(self.raw_func, self._is_method, self._self_param)
            self._python_runner = runner
        return runner.call(instance, args, global_size, local_size, launch=launch)

    # ---------- source generation ----------

    def _source_signature(self)->str:
        """Python source signature: source changes -> regenerate (recompile when cached content differs)."""
        try:
            source:str = inspect.getsource(self.raw_func)
        except OSError:
            source = ""
        return short_hash(f"{source}\nbackend={self.backend}")

    def _resolve_global(self, name:str)->Any:
        """Resolve a name the kernel body reads from outside the function.

        A closure cell takes precedence over a module global, exactly as python resolves
        it, so a nested kernel sees its enclosing function's variable (which is what the
        outer-scope feature parameterizes).
        """
        for cell_name, cell in zip(self.raw_func.__code__.co_freevars,
                                   self.raw_func.__closure__ or ()):
            if cell_name == name:
                try:
                    return cell.cell_contents
                except ValueError:  # the cell is still empty
                    return None
        return self.raw_func.__globals__.get(name)

    def _c_kernel_name(self, sig:str, backend:str)->str:
        return f"pyk_{self.__name__}_{short_hash(sig + '_' + backend)}"

    def _bind_images(self, full_values:Any, backend:str)->None:
        """Create the device objects for every Image argument, before compilation.

        Each backend decides what an image becomes (a texture for reading, a surface
        for writing); the objects are cached on the Image so later launches reuse
        them. Only CUDA needs this ahead of compilation; OpenCL images are created
        per launch without that ordering constraint.
        """
        from ..image import Image

        for value in full_values:
            if not isinstance(value, Image):
                continue
            binding = value._bindings.get(backend)
            if binding is not None:
                continue
            if backend == "cuda":
                value.bind_cuda()
            # opencl binds at launch time; nothing to do here

    def _compile(self, args:List[Any], arg_types:List[Any], self_types:List[Any],
                 self_attrs:List[str], owner:Optional[type], sig:str,
                 backend:str, device_obj:Any, instance:Any=None)->Tuple[Any, List[str]]:
        if backend in ("cuda", "hip"):
            for t in list(arg_types) + list(self_types):
                base:str = (t[1] if isinstance(t, tuple) else t)
                if base == "half":
                    raise SyntaxError(
                        "fp16 (np.float16) arguments are not supported on the cuda backend: "
                        "NVRTC on this machine cannot resolve cuda_fp16.h; use np.float32 instead") from None

        c_name:str = self._c_kernel_name(sig, backend)
        device_cache:Dict[Any, Any] = {}
        self_attr_types:Dict[str, Any] = dict(zip(self_attrs, self_types))

        # hip accepts CUDA syntax (threadIdx/atomicAdd/make_float4 etc.), so codegen goes through cuda
        codegen_backend:str = ("cuda" if backend == "hip" else backend)
        # Statements above the magic loop that only build data are executed by the
        # interpreter, not translated: numpy decides the shape and dtype, and the resulting
        # object is what the kernel receives as a parameter (see
        # FunctionTranslator._host_created_assignment).
        creation_values, _ = _host_created_values(self.raw_func, args, instance)
        translator:FunctionTranslator = FunctionTranslator(
            self.raw_func, list(arg_types) + list(self_types), codegen_backend,
            self._resolve_global, device_cache, is_kernel=True,
            self_param=self._self_param, self_attr_types=self_attr_types,
            self_owner=owner, launch_backend=backend,
            host_creation_values=creation_values)
        translator.translate()
        # One kernel per magic loop: a body with several renders several, launched in order
        # so the arrays stay on the device between them.
        rendered:List[Tuple[str, str]] = translator.render_kernels(c_name)
        kernel_code:str = "\n\n".join(source for _, source in rendered)

        # Device helper functions must be emitted in dependency order: C requires a callee to
        # be declared before its caller, and the cache is filled depth-first (a nested helper
        # is finished *after* the helper that calls it). Emit leaves first.
        dependency_codes:List[str] = [entry["code"] for entry in _dependency_order(device_cache)]

        # Structure types come first, once, before every device function and the kernel: a
        # helper's signature may name one as well, and C++ rejects the same struct defined
        # twice. Collect them from the kernel and from each helper.
        structure_dtypes:List[Any] = list(translator.structure_dtypes())
        for entry in device_cache.values():
            for dtype in entry.get("structs", ()):
                if dtype not in structure_dtypes:
                    structure_dtypes.append(dtype)
        typedefs:str = (structs.render_structures(structure_dtypes, translator.backend)
                        if structure_dtypes else "")

        # Quaternion structures and their arithmetic both come from a fixed header, copied
        # into the cache's include folder beside the generated source. The source needs only
        # the enable macros for the widths in use, then one include -- no declaration of its
        # own, because the header spells the two type names in advance.
        quaternion_needs:Set[str] = set(translator.quaternion_needs)
        for entry in device_cache.values():
            quaternion_needs.update(entry.get("quaternion_needs") or ())
        include_lines:str = ""
        if quaternion_needs:
            from . import quaternion as quaternion_header

            include_lines = quaternion_header.include_line(
                tuple(sorted(quaternion_needs)), backend, self._cache_dir(backend))

        # A nested launch needs the callee's own `__global__` function in this same source:
        # the launch statement names a device function, and C requires it before its caller.
        device_prefix:List[str] = list(translator.nested_helpers) + list(translator.nested_kernels)
        device_prefix.extend(dependency_codes)
        source:str = ("\n".join(device_prefix) + "\n\n" + kernel_code) if device_prefix else kernel_code
        if include_lines:
            # The header declares the types, so it goes after the ordinary typedef block and
            # ahead of every function that names a quaternion.
            source = include_lines + "\n" + source
        if typedefs:
            source = typedefs + "\n" + source

        source_path:str = self._write_source(sig, source, backend)
        linked:bool = translator.nests_a_launch
        kernels:List[Any] = [
            self._backend_compile(source_path, name, backend, device_obj, linked=linked)
            for name, _ in rendered]
        # A phase kernel declares only the shapes it indexes itself, so its launch needs its
        # own trailing extent arguments rather than the ones the first kernel wants.
        extra_shapes:List[List[Tuple[int, int]]] = []
        # The created objects each phase declares: mid = np.zeros(4) between two loops is
        # a parameter of the pass that reads it and of no other.
        phase_created:List[List[str]] = []
        for phase in translator.phase_translators:
            phase_index:Dict[str, int] = {
                name: position for position, (name, _) in enumerate(phase.params)}
            extra_shapes.append([(phase_index[array_name], dim)
                                 for array_name, dim in phase.shapes_used])
            phase_created.append(list(phase.host_values))

        kernel:Any = kernels[0]

        # Atomic targets: static analysis cannot recognize pointer-cast writes, so force them to writable
        from ..constants import KernelFlags
        for target_name in translator.atomic_targets:
            if target_name in kernel._args:
                kernel._args[target_name].kernel_flags = (kernel._args[target_name].kernel_flags | KernelFlags.Writed)

        param_index:Dict[str, int] = {name: i for i, (name, _) in enumerate(translator.params)}
        # A multi-phase body is translated per phase, so the main translator indexes nothing;
        # the first phase's shapes are the ones this launch derives its work size from.
        shapes_owner:Any = (translator.phase_translators[0]
                            if translator.phase_translators else translator)
        shape_index:Dict[str, int] = {
            name: position for position, (name, _) in enumerate(shapes_owner.params)}
        shape_sources:List[Tuple[int, int]] = [
            (shape_index[array_name], dim) for array_name, dim in shapes_owner.shapes_used]
        if not extra_shapes:
            # A single kernel: its own extents are the launch's.
            extra_shapes = [list(shape_sources)]
        # Each entry is (parameter index, dimension) for an array-sourced extent, or
        # (None, count) for a literal one (np.ndindex(2, 3)).
        work_size_dims:List[Tuple[Optional[int], int]] = [
            (shape_index[name], dim) if name else (None, dim)
            for name, dim in shapes_owner.work_size_dims]
        # An explicit gpu.work_size(x) declaration: expand each named array into its own
        # dimensions. The rank is only known at call time, so record (param, -1) markers and
        # let _invoke substitute the real per-dimension indices.
        if not work_size_dims and shapes_owner.declared_work_arrays:
            work_size_dims = [(shape_index[name], -1)
                              for name in shapes_owner.declared_work_arrays]
        # local memory: opencl passes byte sizes at launch by parameter position; cuda uses __shared__ declarations and needs nothing passed
        ctype_sizes:Dict[str, int] = {"bool": 1, "char": 1, "uchar": 1, "short": 2, "ushort": 2,
                                      "int": 4, "uint": 4, "long": 8, "ulong": 8,
                                      "float": 4, "double": 8}
        local_sizes:List[int] = [ctype_sizes.get(ctype, 4) * count for _, ctype, count in translator.local_buffers]
        # Images an OpenCL kernel samples need a sampler argument each; record which
        # parameter supplies the sampler so the launch can append it.
        image_samplers:List[int] = [param_index[name]
                                    for name in sorted(translator.required_image_samplers)]
        # Return slots: one per distinct value type any return site writes, plus the
        # claim/site words. `arg_indices` maps an array a return site names to its
        # position in the launch arguments, so the result can hand back that same object.
        returns:Dict[str, Any] = {
            "slot_types": [translator.return_slot_dtype(index)
                           for index in range(len(translator.return_layout))],
            "slot_kinds": [entry["type"] for entry in translator.return_layout],
            "sites": translator.return_site_layout(),
            "arg_indices": {name: position for name, position in param_index.items()},
        }
        return (kernel, shape_sources, work_size_dims, local_sizes,
                shapes_owner.uses_global_id, image_samplers,
                returns if translator.has_returns else None,
                translator.outer_vars,
                {"param_names": {position: name for name, position in param_index.items()},
                 # Every kernel this body produced, in launch order: one per magic loop.
                 "kernels": kernels,
                 # The trailing extent arguments each phase kernel needs, by phase.
                 "kernel_shapes": extra_shapes,
                 "kernel_created": phase_created,
                 "subscript_indices": dict(translator.subscript_indices),
                 # How many index names the magic loop unpacks: the launch checks it against
                 # the number of dimensions its iteration domain actually has.
                 "loop_names": len(translator.work_item_indices),
                 # The array behind a `np.ndindex(arr.shape)` domain (index into the device
                 # parameter list, or -1), whose rank the loop must name one index per.
                 "whole_shape_index": param_index.get(translator.nd_whole_shape_param, -1)})

    def _cache_dir(self, backend:str)->str:
        source_file:str = getattr(self.raw_func.__code__, "co_filename", "") or ""
        base_dir:str = (os.path.dirname(os.path.abspath(source_file)) if source_file else os.getcwd())
        folder:str = {"opencl": "__clcache__", "cuda": "__cucache__", "hip": "__hipcache__"}[backend]
        return os.path.join(base_dir, folder, "pykernels")

    def _write_source(self, sig:str, source:str, backend:str)->str:
        cache_dir:str = self._cache_dir(backend)
        os.makedirs(cache_dir, exist_ok=True)
        extension:str = {"opencl": "cl", "cuda": "cu", "hip": "hip"}[backend]
        path:str = os.path.join(cache_dir, f"pyk_{self.__name__}_{sig}_{self._source_sig}.{extension}")

        existing:Optional[str] = None
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                existing = file.read()
        if existing != source:
            with open(path, "w", encoding="utf-8", newline="\n") as file:
                file.write(source)
        return path

    def _backend_compile(self, source_path:str, c_name:str, backend:str, device_obj:Any,
                         linked:bool=False):
        if backend == "opencl":
            from ..opencl.oop.program import Program
            program:Any = Program(device_obj.platform.default_context, file_name=source_path)
            return program.kernels[c_name]

        if backend == "hip":
            from ..hip.oop.program import Program as HIPProgram
            from ..hip.oop.hip_compiler import HipCompiler
            device_obj.default_context.make_current()
            compiler = HipCompiler(source_path)
            program:Any = HIPProgram(source_path, compiler=compiler)
            return program.kernels[c_name]

        from ..cuda.oop.program import Program as CUProgram

        device_obj.default_context.make_current()
        if linked:
            # A kernel that launches another kernel needs *linked* device code: the child is
            # resolved through the device runtime, which NVRTC does not link in (and cuLink
            # on Windows will not read cudadevrt.lib). nvcc performs that link, so only such
            # a source pays for it; everything else stays on NVRTC.
            return _compile_linked_cuda(source_path, c_name, device_obj)

        program:Any = CUProgram(source_path)
        return program.kernels[c_name]

    # ---------- launch ----------

    def _build_launch_args(self, args:List[Any], arg_types:List[Any], shape_sources:List[Tuple[int, int]])->List[Any]:
        launch_args:List[Any] = list(args)
        for param_index, dim in shape_sources:
            launch_args.append(int(args[param_index].shape[dim]))
        return launch_args

    def __repr__(self)->str:
        return f"PyKernel({self.__name__}, backend={self.backend})"


def _cuda_toolkit() -> Optional[pathlib.Path]:
    """The CUDA Toolkit directory, or None when nvcc cannot be found."""
    for variable in ("CUDA_PATH", "CUDA_HOME"):
        value:str = os.environ.get(variable, "")
        if value:
            candidate:pathlib.Path = pathlib.Path(value) / "bin" / "nvcc.exe"
            if candidate.exists():
                return candidate
    import shutil

    found:Optional[str] = shutil.which("nvcc")
    return pathlib.Path(found) if found else None


def _cuda_tool(nvcc:pathlib.Path, tool:str) -> Optional[pathlib.Path]:
    """Another executable shipped beside nvcc, or None when it is not installed."""
    candidate:pathlib.Path = nvcc.parent / tool
    if candidate.exists():
        return candidate
    import shutil

    found:Optional[str] = shutil.which(pathlib.Path(tool).stem)
    return pathlib.Path(found) if found else None


class _LinkedImage:
    """A loaded module plus the metadata `Kernel` needs, for an nvcc-linked source.

    `Program` normally owns an NVRTC program and reads the kernel arguments from its
    parser. A linked build comes from nvcc instead, so only the two attributes `Kernel`
    actually touches are provided: the module handle and the per-kernel argument layout,
    which is parsed from the source the same way NVRTC's parser would.
    """

    def __init__(self, module:Any, kernel_infos:Dict[str, Any],
                 context:Any, device:Any)->None:
        self.id = module
        self.kernel_infos:Dict[str, Any] = kernel_infos
        # `Kernel` reaches back into its program for these during a launch, so a linked
        # build has to answer them like an NVRTC one does.
        self.context:Any = context
        self.device:Any = device
        self.prefix:str = "nvcc"


def _compile_linked_cuda(source_path:str, c_name:str, device_obj:Any)->Any:
    """Compile a source that launches a kernel from device code, through nvcc.

    NVRTC cannot produce a loadable image for such a source: the child kernel is resolved
    through the device runtime, which has to be linked in, and the only tool on this
    platform that performs that link is nvcc. Everything else keeps using NVRTC, so the
    cost -- a subprocess and a dependency on the toolkit -- is paid only here.
    """
    import subprocess
    from ctypes import byref, c_void_p, cast, create_string_buffer

    from ..cuda.driver.cuda import CUDA
    from ..cuda.oop.kernel import Kernel
    from ..cuda.oop.program_parser import ProgramParser
    from ..exceptions import CompileError

    nvcc:Optional[pathlib.Path] = _cuda_toolkit()
    if nvcc is None:
        raise SyntaxError(
            "this kernel launches another kernel from device code, which needs CUDA's "
            "dynamic parallelism and therefore a device-code link. NVRTC cannot perform "
            "that link, so nvcc is required: set CUDA_PATH to a CUDA Toolkit install (or "
            "put nvcc on PATH) and try again. Kernels that do not nest a launch keep "
            "working without it")

    arch:str = (f"sm_{device_obj.compute_capability_major}"
                f"{device_obj.compute_capability_minor}")
    image_path:pathlib.Path = pathlib.Path(source_path).with_suffix(".cubin")
    # nvcc compiles `.cu` as C++, where `__global__ void f(...)` gets a mangled symbol, and
    # the host looks the kernel up by the plain name it generated. NVRTC exposes the plain
    # one; nvcc does not, so the entry points are given C linkage here -- only for this
    # build, since a plain NVRTC source must keep the name it already resolves.
    #
    # `-cubin` is not enough: for a source that launches a kernel, nvcc leaves the parent
    # out of the image until the device link runs, and only the *host* link performs that
    # one (a `-cubin` build lists the child alone). So the tools go through a throwaway
    # executable and the device image is extracted from it, which is the one route that
    # produces a parent that `cuModuleGetFunction` can find. That needs a `main`, which the
    # generated source does not have, so the copy gets a stub -- nothing calls it.
    linked_source:pathlib.Path = image_path.with_suffix(".linked.cu")
    text:str = pathlib.Path(source_path).read_text(encoding="utf-8")
    linked_source.write_text(
        text.replace("__global__ void ", 'extern "C" __global__ void ')
        + "\n\nint main() { return 0; }\n", encoding="utf-8")
    executable:pathlib.Path = image_path.with_suffix(".linked.exe")
    command:List[str] = [
        str(nvcc), "-rdc=true", f"-arch={arch}", "-x", "cu",
        "-o", str(executable), str(linked_source),
    ]
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0 or not executable.exists():
        detail:str = (finished.stderr or finished.stdout or "").strip()
        raise CompileError(
            f"nvcc could not build {pathlib.Path(source_path).name} as linked device code "
            f"(it launches a kernel from device code, which needs dynamic parallelism):\n"
            f"{detail[:1200]}")

    extractor:Optional[pathlib.Path] = _cuda_tool(nvcc, "cuobjdump.exe")
    if extractor is None:
        raise CompileError(
            "cuobjdump.exe was not found beside nvcc; it is needed to take the linked device "
            "image out of an nvcc build")
    extract_dir:pathlib.Path = image_path.with_suffix(".linked")
    extract_dir.mkdir(exist_ok=True)
    extracted = subprocess.run(
        [str(extractor), "-xelf", "all", str(executable)],
        capture_output=True, text=True, cwd=str(extract_dir))
    images:List[pathlib.Path] = sorted(extract_dir.glob("*.cubin"))
    if not images:
        detail = (extracted.stderr or extracted.stdout or "").strip()
        raise CompileError(
            f"the linked image could not be extracted from the nvcc build:\n{detail[:900]}")
    image_path.write_bytes(images[0].read_bytes())

    device_obj.default_context.make_current()
    module = c_void_p()
    blob:bytes = image_path.read_bytes()
    buffer = create_string_buffer(blob, len(blob))
    code:int = CUDA.cuModuleLoadData(byref(module), cast(buffer, c_void_p))
    if code != 0:
        raise CompileError(f"the linked image for {pathlib.Path(source_path).name} could "
                           f"not be loaded: {code}")

    # The argument layout comes from the same parser the NVRTC path uses, so the launch
    # binds parameters identically; only the image itself is nvcc's.
    parser:ProgramParser = ProgramParser()
    parser.parse(source_path, None, None, None)
    kernel_infos:Dict[str, Any] = dict(parser.kernel_infos)
    program:Any = _LinkedImage(module, kernel_infos, device_obj.default_context, device_obj)
    kernel:Any = Kernel(program, c_name)
    # `Kernel` starts with an empty argument layout and a `Program` fills it while fetching
    # its kernels; a linked build has no such program, so the parsed layout is handed over.
    kernel.args = kernel_infos[c_name].args
    kernel.type_checked = False
    return kernel


def _check_image_address(image:Any, backend:str)->None:
    """Refuse an address mode this backend cannot honour.

    Only CLAMP is portable. WRAP and MIRROR are defined and implemented in the host
    (python-backend) sampler, but the device samplers read the edge texel instead of
    wrapping: with texel-space coordinates the drivers ignore both CUDA's
    CU_TR_ADDRESS_MODE_WRAP/_MIRROR and OpenCL's CL_ADDRESS_REPEAT/_MIRRORED_REPEAT,
    which was measured on every GPU available while developing this. A launch is refused
    rather than silently returning numbers that differ from the python backend.
    """
    from ..image import ImageAddress

    if image.address is ImageAddress.CLAMP:
        return
    raise ValueError(
        "the %s backend cannot honour address=%r: with texel-space (not normalized) "
        "coordinates the device samplers ignore the wrap and mirror modes and read the "
        "edge texel instead, so this image would not match the python backend. Use "
        "ImageAddress.CLAMP on a GPU backend, or run the kernel on the python backend, "
        "which implements all three modes."
        % (backend, image.address.value))


def _image_sampler(image:Any)->Any:
    """The OpenCL sampler matching an Image's sampling rules.

    OpenCL takes the sampler as a kernel argument while CUDA bakes it into the
    texture object, so this is where the backend-neutral filter/address enums turn
    into an ``sampler_t``. The enum names are the neutral ones, and the mapping here is
    where they become OpenCL's spellings (NEAREST for point, CLAMP_TO_EDGE for clamp).
    """
    from ..image import ImageAddress, ImageFilter
    from ..opencl.driver import (
        cl_addressing_mode,
        cl_filter_mode,
        sampler_t,
    )

    _check_image_address(image, "opencl")

    filter_mode = {
        ImageFilter.POINT: cl_filter_mode.CL_FILTER_NEAREST,
        ImageFilter.LINEAR: cl_filter_mode.CL_FILTER_LINEAR,
    }[image.filter]
    address_mode = {
        ImageAddress.CLAMP: cl_addressing_mode.CL_ADDRESS_CLAMP_TO_EDGE,
    }[image.address]

    # Coordinates are texel-space, matching CUDA and the host path.
    return sampler_t(normalized_coords=False,
                     addressing_mode=address_mode,
                     filter_mode=filter_mode)


def _outer_slot_dtype(t:Any)->Any:
    """The numpy dtype of a one-element buffer carrying a declared outer scalar."""
    return np.dtype(_SCALAR_SLOT_DTYPES[t])


def _scalar_slot_value(value:Any, t:Any)->Any:
    """An outer scalar as the element its buffer stores."""
    if t == "bool":
        return np.uint8(1 if value else 0)
    return value


def _outer_scope_value(raw:Any, t:Any)->Any:
    """A declared outer scalar's buffer element as the python value to store back."""
    if t == "bool":
        return bool(int(raw))
    if t in ("char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"):
        return int(raw)
    return float(raw)


_SCALAR_SLOT_DTYPES:Dict[str, Any] = {
    "bool": np.uint8,
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "long": np.int64,
    "ulong": np.uint64,
    "float": np.float32,
    "double": np.float64,
}


def _check_shape_indices(kernel_name:str, full:List[Any],
                         shape_sources:List[Tuple[int, int]],
                         work_size_dims:List[Tuple[Optional[int], int]],
                         launch_meta:Dict[str, Any])->None:
    """Refuse a dimension the array does not have, in the caller's own terms.

    A magic loop's index count is written in the source (``for i, j, k in ...``) while an
    array's rank is only known from the value passed at call time, so the mismatch can
    only be found here. The python backend reports it as ``not enough values to unpack
    (expected 3, got 2)``; a device launch used to say ``tuple index out of range``,
    which names neither the array nor the loop.

    The same read is applied to how the body subscripts each array: ``out[i, j]`` on a
    3-D array selects a *slice* in python, which a work item cannot represent, and used to
    compile into a single flat element -- a silently wrong answer.
    """
    param_names:Dict[int, str] = launch_meta["param_names"]

    # The highest dimension referenced per array, from the shape arguments the kernel
    # receives and from the work-size derivation of the magic loop.
    highest:Dict[int, int] = {}
    for param_index, dim in shape_sources:
        highest[param_index] = max(highest.get(param_index, -1), dim)
    for param_index, dim in work_size_dims:
        if param_index is not None and dim >= 0:
            highest[param_index] = max(highest.get(param_index, -1), dim)

    for param_index, dim in sorted(highest.items()):
        value:Any = full[param_index]
        shape:Any = getattr(value, "shape", None)
        if shape is None:
            continue
        rank:int = len(shape)
        if dim < rank:
            continue
        name:str = param_names.get(param_index, f"argument {param_index}")
        indexed:int = dim + 1
        raise ValueError(
            f"{kernel_name}(): not enough values to unpack "
            f"(expected {indexed}, got {rank}): '{name}' has shape "
            f"{tuple(int(s) for s in shape)}, so it cannot be indexed with {indexed} "
            f"indices. The magic loop unpacks {indexed} names and must match the number "
            f"of dimensions of the array it iterates"
        ) from None

    name_to_index:Dict[str, int] = {name: index for index, name in param_names.items()}
    for name, count in sorted(launch_meta["subscript_indices"].items()):
        param_index:int = name_to_index.get(name, -1)
        if param_index < 0:
            continue  # a local buffer or an image: no numpy value to measure
        value = full[param_index]
        shape = getattr(value, "shape", None)
        if shape is None:
            continue
        rank = len(shape)
        if count == rank:
            continue
        if count > rank:
            raise ValueError(
                f"{kernel_name}(): too many indices for '{name}': it has shape "
                f"{tuple(int(s) for s in shape)} and is indexed with {count} indices, "
                f"but only {rank} dimension{'s' if rank != 1 else ''} exist"
            ) from None
        raise ValueError(
            f"{kernel_name}(): '{name}' has shape {tuple(int(s) for s in shape)} and is "
            f"indexed with {count} indices; python would select a sub-array there, which "
            f"a work item cannot represent. Index every dimension, e.g. "
            f"'{name}[i, j, k]' for a 3-D array"
        ) from None


def _kernel_scalar_argument(value:Any, t:Any)->Any:
    """A kernel argument in the type the device signature declares for it.

    Only `bool` needs converting: the translator passes it as a byte, because a real
    bool is rejected as a kernel parameter by OpenCL and has no CUDA binding entry. The
    conversion happens on the argument handed to the launch, not on the caller's object,
    so the values a return site hands back are still the caller's own objects.
    """
    if t == "bool":
        return np.uint8(1 if value else 0)
    return value


def _host_created_values(func:Any,
                         args:Optional[Tuple[Any, ...]]=None,
                         instance:Any=None
                         )->Tuple[Dict[str, Any], List[str]]:
    """Build the objects a kernel creates above its magic loop.

    The statements are ordinary python and are run as such, in the function's own
    namespace, so `np.zeros(x.shape[0], dtype=x.dtype)` means exactly what it means
    outside a kernel -- including reading the arguments the caller passed (`args`), which
    is why this runs at call time, not at decoration time. `instance` is the receiver when
    the kernel is a method, so a creation can read `self` (`np.zeros((self.height, ...))`).
    Only the statements the translator recognises as pure creation are executed; everything
    else stays device code.
    """
    from .translator import (_assigned_name_of, _creation_assignments,
                            _creation_target_of, _is_creation_expression)

    values:Dict[str, Any] = {}
    names:List[str] = []
    try:
        assignments:List[Any] = _creation_assignments(func)
    except SyntaxError:
        return values, names
    if not assignments:
        return values, names

    globalns:Dict[str, Any] = dict(func.__globals__)
    localns:Dict[str, Any] = {}
    for arg_name, value in zip(_positional_argument_names(func), args or ()):
        localns[arg_name] = value
    receiver:str = _receiver_name(func)
    if receiver and instance is not None:
        localns[receiver] = instance
    for name, cell in zip(func.__code__.co_freevars, func.__closure__ or ()):
        try:
            localns[name] = cell.cell_contents
        except ValueError:
            pass
    for assignment in assignments:
        target:Optional[str] = _assigned_name_of(assignment)
        if target is None:
            continue
        try:
            value:Any = eval(  # noqa: S307 - the kernel source itself
                compile(ast.Expression(body=assignment.value), "<kernel creation>", "eval"),
                globalns, localns)
        except Exception:
            continue  # the translator reports the real problem, with a source position
        localns[target] = value
        # Only a *creation* becomes a kernel parameter: its object has to reach the device.
        # A plain assignment lifted for the creation's sake stays in the namespace alone --
        # the work items compute it themselves, so shipping it would add an argument nothing
        # reads (and shift every later launch argument by one).
        if _creation_target_of(assignment) is not None and _is_creation_expression(
                assignment.value, func):
            values[target] = value
            names.append(target)
    return values, names


def _receiver_name(func:Any)->str:
    """The name of a method's receiver parameter (`self` / `cls`), or "" for a function."""
    code:Any = getattr(func, "__code__", None)
    if code is None or not code.co_varnames:
        return ""
    first:str = code.co_varnames[0]
    return first if first in ("self", "cls") else ""


def _positional_argument_names(func:Any)->List[str]:
    """The python parameter names of a kernel, skipping the `self` parameter if present."""
    code:Any = getattr(func, "__code__", None)
    if code is None:
        return []
    names:List[str] = list(code.co_varnames[:code.co_argcount])
    if names and names[0] in ("self", "cls"):
        return names[1:]
    return names


def _created_arguments(created:Dict[str, Any])->List[Any]:
    """The launch arguments for the objects a kernel body created, in declaration order."""
    return [_kernel_scalar_argument(value, infer_ctype(value)) for value in created.values()]


def _return_value(raw:Any, dtype:Any, kind:str)->Any:
    """One value slot's stored element as the python value it represents.

    ``kind`` is the translator's type name for the returned position, which is how a
    byte-valued slot is turned back into the bool or the char it came from.
    """
    global _VEC_RETURN_TYPES
    if not _VEC_RETURN_TYPES:
        # The host's own classes: a value slot holds the host layout (a 3-component vector
        # has three components there), which is what the caller gets back.
        from ..vec_types.layout import _host_classes

        for name, cls in _host_classes().items():
            _VEC_RETURN_TYPES.setdefault(str(np.dtype(cls.numpy_dtype())), cls)
    vector:Any = _VEC_RETURN_TYPES.get(str(np.dtype(dtype)))
    if vector is not None:
        return vector.from_buffer_copy(np.asarray(raw).tobytes())
    value:Any = np.asarray(raw).reshape(())[()]
    if isinstance(dtype, np.dtype) and dtype.fields is not None:
        # A structure slot holds a record: the caller gets the record itself, and reads its
        # members exactly as the kernel body did (`result["i"]`).
        return value
    if kind == "bool":
        return bool(int(value))
    if kind in ("char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"):
        return int(value)
    return float(value)


_VEC_RETURN_TYPES:Dict[str, Any] = {}


def _dependency_order(device_cache:Dict[Any, Any])->List[Dict[str, Any]]:
    """Order cached device functions so each callee precedes its callers.

    The cache is populated depth-first, so a nested helper (e.g. `clamp01` called by
    `weight`) is completed after the helper that calls it. Emitting in insertion order
    would then reference a function C has not seen yet. Names are matched textually
    against each entry's generated code.
    """
    entries:List[Dict[str, Any]] = [e for e in device_cache.values() if e.get("status") == "done"]
    if len(entries) <= 1:
        return entries

    # name -> entry, for the callee lookup
    by_name:Dict[str, Dict[str, Any]] = {e["c_name"]: e for e in entries}
    ordered:List[Dict[str, Any]] = []
    visiting:set = set()
    done:set = set()

    def visit(entry:Dict[str, Any]) -> None:
        name:str = entry["c_name"]
        if name in done or name in visiting:
            # `visiting` breaks cycles: mutual recursion is rejected elsewhere anyway.
            return
        visiting.add(name)
        for other_name in by_name:
            if other_name != name and other_name in entry["code"]:
                visit(by_name[other_name])
        visiting.discard(name)
        done.add(name)
        ordered.append(entry)
    for entry in entries:
        visit(entry)
    return ordered


def kernel(func:Optional[Callable]=None, *, backend:Optional[str]=None):
    """The `@kernel` decorator.

    When backend is not specified, probe in DEFAULT_BACKEND_ORDER (cuda -> hip -> opencl)
    and take the first backend that enumerates at least one device; the result is cached in-process.

    Usage::

        from pygpgpu.kernel import kernel
        import numpy as np

        @kernel
        def add(a, b, out):
            for i in range(out.shape[0]):
                out[i] = a[i] + b[i]

        add(np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) * 2, out)

    - The outermost `for i in range(<array parameter>.shape[0])` loop is the work-item loop;
      the work size is taken automatically from that array's first dimension;
    - plain python functions called by the kernel are translated into device functions (monomorphized per argument types);
    - helpers may be reached by name: a bare module-level name, `module.func(...)`, `SomeClass.static_method(...)`,
      `self.static_method(...)`, or `self.other_method(...)` for another method of the same class;
    - math.* and abs/min/max/round map to device builtins;
    - unsupported syntax raises SyntaxError.
    """
    def decorator(f:Callable)->PyKernel:
        if backend is not None:
            return PyKernel(f, backend, explicit_backend=True)
        # No backend named: the call-time default decides, so this snapshot is only used
        # for reporting until set_default_device() (or a bracket) has a say.
        return PyKernel(f, _default_backend(), explicit_backend=False)

    if func is not None:
        return decorator(func)
    return decorator
