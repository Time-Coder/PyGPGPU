from __future__ import annotations

import asyncio
import ast
import concurrent.futures
import functools
import inspect
import os
import pathlib
import sys
import threading
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    overload,
)

import numpy as np

from typing_extensions import ParamSpec

from ..vec_types.layout import pack
from ..utils import local_size_for_global, detect_work_size, device_limits, blocks_to_global
from .python_backend import PythonKernel, analyse
from . import structs
from ..numpy import ndarray as gnp
from .translator import (
    PTR,
    DeviceType,
    FunctionTranslator,
    _assigned_name_of,
    _unpack_count_message,
    infer_ctype,
    is_ptr,
    scan_self_attrs,
    short_hash,
    type_sig,
    validate_type_map,
)
from .objects import (ObjectBinding, _complex_member_value,  # noqa: PLC2701 (shared
                    _is_complex_member_class, convert_object_arguments)
from . import source_paths


# ---------- typing hooks ----------

if TYPE_CHECKING:
    # `Image` is imported inside the functions that use it (it is a heavy module and this
    # package keeps backend imports lazy), so an annotation needs the name here instead.
    from ..image import Image

# A kernel is a callable object, so the IDE's parameter hints come from its type, not from
# the ``__call__`` of the class it happens to be. ``PyKernel`` is generic over the decorated
# function's own parameters (PEP 612), which is what lets a type checker see
# ``add(a, b, out)``'s signature on ``add`` itself. Before this, ``@kernel`` published
# ``PyKernel`` as its return type and the decorated name was read as the *decorator*
# (``(f) -> PyKernel``), so every call showed ``__call__(*args, **kwargs)`` and lost its
# parameter list.
#
# The generic base is the one construct that differs by interpreter: a user-defined generic
# over a ParamSpec needs Python 3.10+, and this package supports 3.7+, so the base exists
# only for the type checkers; at runtime it is a plain class whose ``__class_getitem__``
# returns itself. Subscripting ``PyKernel[P]`` therefore never fails at import.
_P = ParamSpec("_P")
_R = TypeVar("_R")

if TYPE_CHECKING:
    class _KernelBase(Generic[_P]):
        """``Generic`` over the decorated function's parameters, for the type checkers."""
else:
    class _KernelBase:
        """Runtime stand-in: subscripting returns this class, so ``PyKernel[_P]`` imports."""
        def __class_getitem__(cls, item:object)->type:
            return cls


if TYPE_CHECKING:
    # `TypedDict` is 3.8+ in typing; this package supports 3.7, so it comes from
    # the backport the rest of the package already uses (see ts_parser.py).
    from typing_extensions import TypedDict

    from .translator import ReturnSiteLayout

    # ---------- launch records ----------

    # What `_bind_outer_values` hands the launch: the arguments carrying this call's outer-scope
    # values, the declared outer scalars whose final value is written back, and the full
    # argument list those values prefix.
    class WritableOuter(TypedDict):
        name: str
        kind: str
        buffer: "np.ndarray[Any, Any]"
        type: DeviceType


    class OuterState(TypedDict):
        args: List[object]
        writable: List[WritableOuter]
        full: List[object]


    # The device-side slots a body's returns need: one buffer per value slot, plus the shared
    # claim word (which work item won) and the site word (which `return` it was).
    class ReturnState(TypedDict):
        buffers: List["np.ndarray[Any, Any]"]
        claim: "np.ndarray[Any, Any]"
        site: "np.ndarray[Any, Any]"


    # What the translator records about the returns themselves: the dtype of each value slot,
    # and the per-site layout that says how to rebuild the result.
    class ReturnSlots(TypedDict):
        slot_types: List["np.dtype[Any]"]
        slot_kinds: List[str]
        arg_indices: Dict[str, int]
        sites: List[ReturnSiteLayout]

# ---------- bracket selector parsing ----------

_BACKEND_ALIASES:Dict[str, str] = {
    "opencl": "opencl", "ocl": "opencl", "cl": "opencl",
    "cuda": "cuda", "cu": "cuda",
    "hip": "hip", "rocm": "hip", "amd": "hip",
    "metal": "metal", "mtl": "metal",
    # The pure-python backend: no translation, no device, serial execution on the host.
    "python": "python", "py": "python", "host": "python",
}

# Probe order when no backend is specified: take the first backend that enumerates at least one device
DEFAULT_BACKEND_ORDER:Tuple[str, ...] = ("cuda", "hip", "metal", "opencl")

_probed_backend:Optional[str] = None

# The selector list a bracketed-less call walks, set by set_default_device(). None means
# "use DEFAULT_BACKEND_ORDER". Each entry is (backend, device-or-None).
_default_devices:Optional[List[Tuple[Optional[str], Optional[str]]]] = None
_default_device_lock = threading.Lock()

# The process-wide `int`/`float` mapping `set_type_map()` installs, or None. Read **once**,
# when a kernel is decorated: a mapping is part of what that kernel means, so changing it
# afterwards cannot change a kernel that already exists (its `type_map` is a snapshot, and it
# is in its cache key). That is also what keeps `set_type_map()` honest across a warm cache.
_type_map_default:Optional[Dict[str, str]] = None
_type_map_lock = threading.Lock()


def set_type_map(int:object=None, float:object=None)->None:
    """Set the default device width a python ``int`` / ``float`` travels as.

    Called with no arguments it restores the built-in widths (``int`` = 32-bit ``int``,
    ``float`` = 32-bit ``float``)::

        set_type_map(float=np.float64)     # every kernel decorated from now on is 64-bit

        @kernel
        def accurate(scale, out): ...       # `scale` is a `double`

    The values are the same type objects `@kernel(float=...)` takes -- a numpy dtype or
    scalar type, a ctypes scalar class, or a device type name. A decorator's own `int=`/
    `float=` wins over this, and a numpy scalar keeps its own width either way.

    **It applies at decoration time, not at call time.** A kernel that already exists keeps
    the mapping it was declared with, including across a warm cache and across processes:
    the mapping is in the generated source's digest and in the C name, so a later setting
    cannot be served the binary an earlier one compiled. Call it before decorating.
    """
    global _type_map_default
    if int is None and float is None:
        with _type_map_lock:
            _type_map_default = None
        return
    declared:Dict[str, object] = {}
    if int is not None:
        declared["int"] = int
    if float is not None:
        declared["float"] = float
    resolved:Dict[str, str] = validate_type_map(declared)
    with _type_map_lock:
        _type_map_default = resolved


def get_type_map()->Dict[str, str]:
    """The process-wide mapping `set_type_map()` installed, as device type names."""
    with _type_map_lock:
        current:Optional[Dict[str, str]] = _type_map_default
    return dict(current) if current else {}


def _effective_type_map(type_map:Optional[Dict[str, object]])->Dict[str, str]:
    """The mapping a kernel is declared with: its own, over the process-wide default."""
    base:Dict[str, str] = get_type_map()
    if not type_map:
        return base
    resolved:Dict[str, str] = validate_type_map(type_map)
    base.update(resolved)
    return base


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


def _backend_devices(backend:str)->List[object]:
    """Enumerate the backend's device list (order = numbering used by plain-numeric selectors); raises when the runtime is missing."""
    if backend == "opencl":
        from ..opencl.oop.platforms import Platforms
        return [d for platform in Platforms for d in platform.devices]
    if backend == "hip":
        from ..hip.oop.devices import Devices
        return list(Devices)
    if backend == "metal":
        from ..metal.oop.devices import Devices
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


def _parse_work_item(item:object)->Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """Parse a work-size bracket, in CUDA's own terms: ``(blocks, threads)``.

    The bracket states a launch the way a CUDA or HIP programmer states one -- how many
    blocks per grid and how many threads per block -- because that is the spelling those
    readers already use, and OpenCL users who want a work group write ``local=`` in their
    own API rather than a bracket. So ``k[(4, 256)]`` is 4 blocks of 256 threads: 1024 work
    items. ``k[(n,)]`` / ``k[n]`` is a plain item count with an automatic group size.

    What travels onwards is the *total work-item* form every backend already speaks
    (``global`` items split by a ``local`` group size), computed here as
    ``blocks * threads``. Doing the conversion at this one entry point keeps every
    downstream rule unchanged: the geometry a CUDA/HIP launch derives stays
    ``global // local``, which is now ``blocks`` by construction, and the OpenCL launch
    keeps receiving the total size its API wants. It also removes a check that only the
    old spelling needed: a total size could fail to divide by the group size, while a block
    count multiplied by a thread count always divides (the quotient *is* the block count).

    ``None`` still means "automatic". A one-element bracket is a bare item count, so it is
    the same number under either spelling and needs no conversion.
    """
    def norm(v:object)->Optional[Tuple[int, ...]]:
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

    def blocks_and_threads(blocks:Optional[Tuple[int, ...]],
                           threads:Optional[Tuple[int, ...]]
                           )->Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
        """``(blocks, threads)`` -> the ``(global, local)`` form the backends speak."""
        if blocks is None or threads is None:
            # One half alone: `threads` alone has no meaning (a group size with no grid),
            # and `blocks` alone keeps the old "count with an automatic group" reading,
            # which is the same number either way.
            return blocks, threads
        return blocks_to_global(blocks, threads)

    if isinstance(item, int):
        return norm(item), None
    if isinstance(item, tuple):
        if len(item) == 1:
            return norm(item[0]), None
        if len(item) == 2:
            return blocks_and_threads(norm(item[0]), norm(item[1]))
    raise TypeError(
        "work size bracket must be n, (blocks,), or (blocks, threads) -- a CUDA-style "
        "launch; use None for an automatic size")


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


def _device_match_keys(sel:str, device:object)->bool:
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


def _resolve_device(backend:str, device_sel:Optional[str])->Tuple[object, str]:
    """Resolve a device within a backend: None/index/name substring ("name:ordinal" disambiguates); returns (device object, cache key)."""
    devices:List[object] = _backend_devices(backend)

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

    matches:List[object] = [d for d in devices if _device_match_keys(sel, d)]
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
            devices:List[object] = _backend_devices(backend)
        except Exception:
            continue
        for i in range(len(devices)):
            selectors.append(f"{backend}:{i}")
    return selectors


# The name this function had before; kept so existing call sites keep working.
available_devices = list_devices


def get_device(selector:str)->object:
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


def _bind_arguments(func:Callable, args:Tuple[object, ...],
                    kwargs:Dict[str, object],
                    skip_first:bool=False)->Tuple[object, ...]:
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

    ordered:List[object] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in bound.arguments:
            ordered.append(bound.arguments[name])
        else:
            # bind() accepted the call without it, so the parameter has a default.
            ordered.append(parameter.default)
    return tuple(ordered)


def _bound_arguments_for_call(func:Callable, args:Tuple[object, ...],
                              kwargs:Dict[str, object],
                              skip_first:bool=False)->Tuple[object, ...]:
    """``_bind_arguments`` as the entry points call it (see _native_argument_errors)."""
    return _bind_arguments(func, args, kwargs, skip_first)


def _native_argument_errors(entry:Callable[_P, _R])->Callable[_P, _R]:
    """Give an entry point native-python's single-frame traceback for a bad call.

    Native python blames the call site and nothing else; a decorated call would otherwise
    also show this module's ``__call__`` and the signature stub, so the same mistake looked
    different once a function was decorated. The trimming happens here, at the entry point,
    because clearing the traceback deeper inside and re-raising lets every frame the
    exception travels through push itself back on again.

    The parameters are forwarded unchanged (``_P``/``_R``), so wrapping an entry point does
    not hide its signature from a type checker: ``functools.wraps`` alone conservatively
    reports ``(*args, **kwargs)``, which would undo the parameter hints the caller's own
    ``def`` gave the kernel.

    A failure of the *body* travels through here too, and arrives already carrying a
    stand-in for the body's own line (see ``PyKernel._invoke``). The wrapper is the last
    place its own frame can be dropped, because a bare ``raise`` here would push it back on:
    only the stand-in is kept, so the reader sees the caller's line and the body's line and
    nothing of this library.
    """
    @functools.wraps(entry)
    def wrapper(*args:_P.args, **kwargs:_P.kwargs)->_R:
        try:
            return entry(*args, **kwargs)
        except (ArgumentTypeError, TypeError) as error:
            error.__traceback__ = None
            raise
        except Exception as error:
            body:Optional[TracebackType] = _body_only(error)
            if body is None:
                raise
            error.__traceback__ = body
            raise

    return wrapper


def _body_only(error:BaseException)->Optional[TracebackType]:
    """The exceptions raised by a kernel body, without this library's frames.

    ``PyKernel._invoke`` ends the chain with the body's own frame: a stand-in on a device
    backend, and the *real* frame on the interpreter backend, which has column information
    a stand-in cannot carry. Everything ahead of it belongs to the launch machinery.
    Nothing is returned for an error that carries neither -- a failure raised before the
    body ran -- so that one keeps its own diagnostic.
    """
    traceback:object = error.__traceback__
    keep:object = None
    cursor:object = traceback
    while cursor is not None:
        code:object = cursor.tb_frame.f_code
        if _is_body_stand_in(code) or _is_user_frame(code):
            # The last such frame is the body's: the launch machinery is all inside this
            # package, so anything after it would be a callee, and the body is the callee.
            keep = cursor
        cursor = cursor.tb_next
    return keep


def _is_user_frame(code:object)->bool:
    """True for a frame of code that lives outside this package."""
    filename:str = getattr(code, "co_filename", "") or ""
    if not filename:
        return False
    if filename.startswith("<") and filename.endswith(">"):
        # `-c`, stdin and exec'd code: no library of ours is written that way.
        return True
    package_dir:str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return not os.path.abspath(filename).startswith(package_dir + os.sep)


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
_RECEIVER_PLACEHOLDER:object = object()


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
    namespace:Dict[str, object] = {"__name__": "pygpgpu.kernel._signature_stub"}
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
    annotations:object = getattr(func, "__annotations__", None) or {}
    annotation:object = annotations.get(name, None)
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


def _check_annotated_argument(func:Callable, name:str, value:object, qualname:str) -> None:
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
                     stub:Callable, args:Tuple[object, ...],
                     kwargs:Dict[str, object], skip_first:bool)->inspect.BoundArguments:
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


def _caller_traceback(traceback:Optional[TracebackType])->Optional[TracebackType]:
    """The first frame of ``traceback`` that is not inside this package."""
    if traceback is None:
        return None
    package_dir:str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prefix:str = package_dir + os.sep
    cursor:object = traceback
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

    keep:object = None
    cursor:object = traceback
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


# A comment constant that marks the frame stand-in built by _body_frame, so the entry
# wrapper can recognize a body failure among the errors it forwards.
_FRAME_MARKER = "__pygpgpu_body_frame__"


def _is_body_stand_in(code:object)->bool:
    """True for the code object :func:`_body_frame` compiles.

    The stand-in is the only function this library creates inside the user's own file, so
    its presence in a traceback identifies a body failure -- and it is the frame the
    caller's own python would have shown. The marker is a constant in the compiled body,
    which no ordinary function carries.
    """
    for constant in getattr(code, "co_consts", ()) or ():
        if isinstance(constant, str) and constant == _FRAME_MARKER:
            return True
    return False


def _subscript_stand_in(error:BaseException, filename:str, lineno:int, column:int,
                        func_name:str)->Optional[Tuple[str, Dict[str, object]]]:
    """A stand-in that really performs the too-wide subscript.

    CPython draws a caret from the failing *instruction*'s position, and the only way to get
    the one python would draw -- the object under `~`, the index under `^` -- is for the
    failing instruction to be the same subscript. So this builds a function whose last
    statement is the user's own expression, with the array bound to one of the real rank so
    numpy raises the identical ``IndexError``.

    Returns ``(source, namespace)``, or None when the recorded expression cannot be
    reproduced -- a target that is not a plain name (``self.items[i]``), or a slice whose
    own names cannot be bound -- in which case the caller falls back to a plain raise.
    """
    target:str = str(getattr(error, "subscript_text", "") or "")
    rank:int = int(getattr(error, "array_rank", 0) or 0)
    if not target or rank <= 0 or "[" not in target or not target.endswith("]"):
        return None
    base:str = target.split("[", 1)[0]
    if not base.isidentifier():
        return None
    try:
        parsed:ast.AST = ast.parse(target, mode="eval")
    except SyntaxError:
        return None
    # Every name the expression mentions stands for an axis; the array itself gets a shape
    # of the real rank, which is what makes numpy refuse the subscript with python's own
    # words and mark the position CPython would.
    namespace:Dict[str, object] = {node.id: 0 for node in ast.walk(parsed)
                                  if isinstance(node, ast.Name)}
    namespace[base] = np.zeros((1,) * rank, dtype=np.float32)
    padding:str = " " * max(column, 0)
    source:str = ("\n" * max(lineno - 3, 0)
                  + f"def {func_name}():\n"
                  + padding + f"{_FRAME_MARKER!r}\n"
                  + padding + target + " = 1.0\n")
    return source, namespace


def _body_frame(func:Callable, error:BaseException)->Optional[TracebackType]:
    """A frame standing in for ``func``'s own body at the line that failed.

    Native python would show exactly two: the caller's line and the failing line inside the
    function. The body's real frame cannot simply be kept, because it is a frame of the
    interpreter backend's machinery rather than of the user's code -- and the device
    backends never execute the body at all.

    A traceback frame's ``co_filename``/``tb_lineno`` are read-only, so the stand-in is a
    function compiled from source carrying the user's filename and enough blank lines to
    place its ``raise`` on the failing line. Its code positions then supply the same caret
    CPython would draw. Only the mapping matters: the file, line and column are what a
    reader sees.
    """
    filename:str = getattr(getattr(func, "__code__", None), "co_filename", "") or ""
    if not filename or filename.startswith("<"):
        return None
    # The line the error points at, in the user's file. For an unpacking failure that is the
    # `for` statement itself, which is exactly what python blames. A *device* backend refuses
    # that shape before the body ever runs, so there is no frame to read it from -- the
    # translator records the position on the exception instead (see _unpacking_error).
    qualname_for_frame:str = (getattr(func, "__qualname__", None)
                              or getattr(func, "__name__", "kernel"))
    body_name:str = qualname_for_frame.rsplit(".", 1)[-1]

    lineno:int = int(getattr(error, "lineno", 0) or 0)
    if not lineno:
        cursor:object = error.__traceback__
        while cursor is not None:
            frame_code:object = cursor.tb_frame.f_code
            if (os.path.abspath(frame_code.co_filename) == os.path.abspath(filename)
                    and frame_code.co_name == body_name
                    and not _is_body_stand_in(frame_code)):
                # The interpreter backend ran the body, so this is the user's own frame at
                # the failing line -- with the column information only a real frame has.
                # Nothing is gained by replacing it with a stand-in, and the caret is lost
                # by doing so, which is what made this backend's error read differently
                # from the undecorated one.
                while cursor.tb_next is not None:
                    cursor = cursor.tb_next
                return cursor
            cursor = cursor.tb_next
        cursor = error.__traceback__
        while cursor is not None:
            if os.path.abspath(cursor.tb_frame.f_code.co_filename) == os.path.abspath(filename):
                lineno = cursor.tb_lineno
                break
            cursor = cursor.tb_next
    if not lineno:
        return None

    qualname:str = getattr(func, "__qualname__", None) or getattr(func, "__name__", "kernel")
    # A nested function's qualname carries its outer names; the frame shows the last one.
    func_name:str = qualname.rsplit(".", 1)[-1]
    # The caret's start column: what python underlines is the offending expression, and the
    # translator records where it begins (see `_check_shape_indices`). A line-only message
    # would point at the statement with no indication of *which* part of it failed.
    column:int = max(int(getattr(error, "column", 0) or 0), 0)
    # The statements need the block's own four spaces *plus* the caret's column, so an
    # error with no recorded column still compiles as an ordinary indented body.
    padding:str = " " * (column + 4)
    # The marker is a real string CONSTANT in the code object, which is what identifies this
    # stand-in later (see _is_body_stand_in). It sits on the line *above* the raise so the
    # raise stays on `lineno`; a comment would not reach `co_consts`.
    source:str = ("\n" * max(lineno - 3, 0)
                  + f"def {func_name}():\n"
                  + padding + f"'{_FRAME_MARKER}'\n"
                  + padding + "raise _PYGPGPU_FRAME_EXCEPTION\n")
    # Prefer a stand-in that performs the real subscript, because that is the only way to
    # get CPython's own caret; the plain raise below is the fallback.
    reproduced:Optional[Tuple[str, Dict[str, object]]] = _subscript_stand_in(
        error, filename, lineno, column, func_name)
    if reproduced is not None:
        source, namespace = reproduced
    else:
        namespace = {"_PYGPGPU_FRAME_EXCEPTION": error}
    try:
        exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - a frame stand-in
        frame_func:object = namespace[func_name]
    except (SyntaxError, ValueError):
        return None
    try:
        frame_func()
    except BaseException:  # noqa: BLE001 - the carrier raises; the real error travels along
        # Only the frame for the user's file is wanted: the caller's frame comes from
        # `_caller_traceback` and everything else in this chain is this module's own.
        cursor = sys.exc_info()[2]
        while cursor is not None:
            if os.path.abspath(cursor.tb_frame.f_code.co_filename) == os.path.abspath(filename):
                # The captured chain continues into this function and its caller, which are
                # library frames -- the stand-in is a whole traceback, not just its head.
                cursor.tb_next = None
                return cursor
            cursor = cursor.tb_next
    return None


def _user_traceback(error:BaseException, func:Callable)->Optional[TracebackType]:
    """The traceback native python would have produced: the caller, then the failing line.

    Everything between the two belongs to this library, so it is spliced out. The head is
    the caller's own frame -- the line that asked the kernel to run -- and its successor is
    the stand-in built by :func:`_body_frame`, which carries the body's file, line and
    function name.

    CPython prepends the frame that *executes* the ``raise``, so the chain cannot be
    assembled by assigning ``tb_next``: the caller node is already part of the live chain,
    and pointing it at the stand-in is rejected as a loop. It is assembled the way the
    interpreter assembles one -- raise once from the stand-in, then drop that one frame --
    which leaves exactly the frames of a native failure.
    """
    body:Optional[TracebackType] = _body_frame(func, error)
    if body is None:
        return _caller_traceback(error.__traceback__)
    body.tb_next = None
    try:
        # Never returns: the stand-in raises the caller's own exception object.
        raise error.with_traceback(body)
    except BaseException as raised:
        # `raised.__traceback__` is this function's own frame followed by the stand-in.
        head:object = raised.__traceback__
        return head.tb_next if head is not None and head.tb_next is not None else body


class _KernelCall(_KernelBase[_P]):
    """Bracket-configuration form: k["cuda"][(64, 8)](args) / k[(64, 8)]["cuda"](args), any order.

    Generic over the decorated function's parameters: a bracket only chooses a backend, a
    device or a work size, so the call it finally makes has the kernel's own signature --
    which is what lets ``k["cuda"](a, out)`` suggest ``a``/``out`` just as ``k(a, out)`` does.
    """

    def __init__(self, kernel:"PyKernel[_P]", instance:object,
                 backend:Optional[str]=None, device:Optional[str]=None,
                 global_size:Optional[Tuple[int, ...]]=None, local_size:Optional[Tuple[int, ...]]=None):
        self._kernel:"PyKernel[_P]" = kernel
        self._instance:object = instance
        self._backend:Optional[str] = backend
        self._device:Optional[str] = device
        self._global_size:Optional[Tuple[int, ...]] = global_size
        self._local_size:Optional[Tuple[int, ...]] = local_size

    def __getitem__(self, item:object)->"_KernelCall[_P]":
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
    # The result is the kernel body's value, produced on the device. `_R` is the
    # decorated function's result type but is not bound on this class, so the
    # honest annotation is the untyped one -- `object` would forbid unpacking a
    # tuple return at the call site.
    def __call__(self, *args:_P.args, **kwargs:_P.kwargs)->Any:
        instance:object = self._instance
        instance_arg:Optional[object] = None
        if self._kernel._is_method and instance is None:
            # class form: Klass.method["cuda"](instance, args...)
            if not args:
                raise TypeError(
                    f"{self._kernel.__name__}() is an instance method; pass the instance "
                    f"as the first argument")
            instance_arg = args[0]
            args = args[1:]
            instance = instance_arg
        call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=self._kernel._is_method)
        return self._kernel._invoke(
            instance, call_args, self._backend, self._device, self._global_size, self._local_size)

    def _launch(self, launch:str, args:Tuple[object, ...],
                kwargs:Optional[Dict[str, object]]=None)->object:
        """Shared body of the three launch styles, honouring the bracket selectors."""
        instance:object = self._instance
        if self._kernel._is_method and instance is None:
            if not args:
                raise TypeError(
                    f"{self._kernel.__name__}() is an instance method; pass the instance "
                    f"as the first argument")
            instance = args[0]
            args = args[1:]
        call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args,
                                                    kwargs or {},
                                                    skip_first=self._kernel._is_method)
        return self._kernel._invoke(
            instance, call_args, self._backend, self._device,
            self._global_size, self._local_size, launch=launch)

    @_native_argument_errors
    def submit(self, *args:object, **kwargs:object)->concurrent.futures.Future:
        """Non-blocking launch of the bracketed form, e.g. ``k["cuda"].submit(...)``."""
        return self._launch("submit", args, kwargs)

    @_native_argument_errors
    def async_call(self, *args:object, **kwargs:object)->asyncio.Future:
        """Non-blocking asyncio launch of the bracketed form, e.g. ``k["cuda"].async_call(...)``."""
        return self._launch("async", args, kwargs)


class _BoundKernelMethod(_KernelBase[_P]):
    """Bound instance-method form: instance.kernel_method(...)

    Reached through ``PyKernel.__get__``. Its parameters are deliberately loose: the
    decorated method's ``P`` still carries the receiver, which a type checker has no way to
    drop from a descriptor's return type, so spelling ``P`` here would make every correct
    ``obj.method(x, y)`` look like ``method(self=x, x=y)`` and report a missing argument.
    The receiver is also the one thing a caller never passes explicitly -- the class form
    ``Klass.method(obj, x, y)`` keeps its full signature, because that access goes through
    ``PyKernel`` itself.
    """

    def __init__(self, kernel:"PyKernel[_P]", instance:object):
        self._kernel:"PyKernel[_P]" = kernel
        self._instance:object = instance

    def __getitem__(self, item:object)->object:
        return _KernelCall(self._kernel, self._instance)[item]

    @_native_argument_errors
    # The result is the kernel body's value, produced on the device. `_R` is the
    # decorated function's result type but is not bound on this class, so the
    # honest annotation is the untyped one -- `object` would forbid unpacking a
    # tuple return at the call site.
    def __call__(self, *args:object, **kwargs:object)->Any:
        call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=True)
        return self._kernel._invoke(self._instance, call_args, None, None, None, None)

    @_native_argument_errors
    def submit(self, *args:object, **kwargs:object)->concurrent.futures.Future:
        """Non-blocking launch: ``instance.kernel_method.submit(...)``."""
        call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=True)
        return self._kernel._invoke(self._instance, call_args, None, None, None, None,
                                    launch="submit")

    @_native_argument_errors
    def async_call(self, *args:object, **kwargs:object)->asyncio.Future:
        """Non-blocking asyncio launch: ``instance.kernel_method.async_call(...)``."""
        call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=True)
        return self._kernel._invoke(self._instance, call_args, None, None, None, None,
                                    launch="async")


class PyKernel(_KernelBase[_P]):
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

    The class is generic over the decorated function's own parameters (PEP 612), which is what
    gives an editor the caller's parameter names on a published kernel -- see _KernelBase above.
    """

    def __init__(self, func:Callable[..., Any], backend:str, explicit_backend:bool=False,
                 domain:Optional[str]=None,
                 type_map:Optional[Dict[str, object]]=None):
        if isinstance(func, classmethod):
            raise SyntaxError("@kernel cannot decorate classmethods (cls semantics are host-only)") from None
        if isinstance(func, staticmethod):
            func = func.__func__

        self.raw_func:Callable = func
        self.backend:str = backend
        # True when the decorator named a backend, so set_default_device() must not
        # override it. A backend chosen by probing is only a snapshot, not a binding.
        self._explicit_backend:bool = explicit_backend
        # The parameter whose shape declares the iteration domain, or None. A gid-style body
        # has no magic loop to read a domain from, and naming the array here says so as part
        # of the signature rather than as a statement inside the body that the device never
        # sees.
        self._domain:Optional[str] = domain
        # The device width a python `int`/`float` travels as, resolved to device type names
        # when the decorator ran and **not** read again later: this decorator's mapping is
        # part of what this kernel *is*, so a later set_type_map() must not change kernels
        # that were already declared. Only the two builtins are here; a numpy scalar names
        # its own width and keeps it (see translator.infer_ctype).
        self._type_map:Dict[str, str] = _effective_type_map(type_map)
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

        self._cache:Dict[str, Tuple[object, List[Tuple[int, int]], List[Tuple[int, int]], List[int]]] = {}
        # The python backend executes the body as-is, so there is nothing to specialize or
        # cache: one runner per kernel object serves every call.
        self._python_runner:object = None
        self._source_sig:str = self._source_signature()

    @overload
    def __get__(self, obj:None, objtype:Optional[type]=None)->"PyKernel[_P]": ...
    @overload
    def __get__(self, obj:object, objtype:Optional[type]=None)->_BoundKernelMethod[_P]: ...
    def __get__(self, obj:object, objtype:Optional[type]=None)->object:
        # descriptor protocol: instance access returns the bound form, class access/static methods return self
        if obj is None or not self._is_method:
            return self
        return _BoundKernelMethod(self, obj)

    def __getitem__(self, item:object)->_KernelCall[_P]:
        return _KernelCall(self, None)[item]

    @_native_argument_errors
    # The result is the kernel body's value, produced on the device. `_R` is the
    # decorated function's result type but is not bound on this class, so the
    # honest annotation is the untyped one -- `object` would forbid unpacking a
    # tuple return at the call site.
    def __call__(self, *args:object, **kwargs:object)->Any:
        if self._is_method:
            if not args:
                raise TypeError(
                    f"{self.__name__}() is an instance method; call it on an instance "
                    f"or pass the instance as the first argument")
            call_args:Tuple[object, ...] = _bind_arguments(self.raw_func, args[1:], kwargs, skip_first=True)
            return self._invoke(args[0], call_args, None, None, None, None)
        call_args = _bind_arguments(self.raw_func, args, kwargs)
        return self._invoke(None, call_args, None, None, None, None)

    @_native_argument_errors
    def submit(self, *args:object, **kwargs:object)->concurrent.futures.Future:
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
            call_args:Tuple[object, ...] = _bind_arguments(self.raw_func, args[1:], kwargs, skip_first=True)
            return self._invoke(args[0], call_args, None, None, None, None, launch="submit")
        call_args = _bind_arguments(self.raw_func, args, kwargs)
        return self._invoke(None, call_args, None, None, None, None, launch="submit")

    @_native_argument_errors
    def async_call(self, *args:object, **kwargs:object)->asyncio.Future:
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
            call_args:Tuple[object, ...] = _bind_arguments(self.raw_func, args[1:], kwargs, skip_first=True)
            return self._invoke(args[0], call_args, None, None, None, None, launch="async")
        call_args = _bind_arguments(self.raw_func, args, kwargs)
        return self._invoke(None, call_args, None, None, None, None, launch="async")

    def _invoke(self, instance:object, args:Tuple[object, ...],
                backend_sel:Optional[str], device_sel:Optional[str],
                global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                launch:str = "call")->object:
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
        except ValueError as caught:
            # Two different things arrive as a ValueError, and python's own reading decides
            # which treatment each takes.
            #
            # An *argument* refusal -- an integer too wide for the type map (`_check_integer_fits`)
            # -- is about the call, not the body: it carries no source position, and it is
            # trimmed to the caller's frame the way a SyntaxError is. Falling through to the
            # body handler would dress it up as a failure inside the function.
            #
            # One the translator *positioned* (see `_position_error`) is a failure of the
            # body's own line -- an unpacking count that cannot match, a subscript that is
            # too wide -- and reads exactly as the interpreter reads it, so it gets the body
            # treatment below: the caller's line and then the line that failed.
            if not str(caught):
                raise
            if getattr(caught, "lineno", None):
                caught.__traceback__ = _user_traceback(caught, self.raw_func)
                raise caught from None
            caller = _caller_traceback(caught.__traceback__)
            raise caught.with_traceback(caller) from None
        except Exception as caught:
            # A failure of the *body* -- a wrong unpack, an out-of-range index -- has to read
            # exactly as it does without the decorator: the caller's line, then the line in
            # the function that failed. The body's own frame belongs to the execution
            # machinery, and the device backends never run the body at all, so it is
            # replaced by a stand-in carrying the same file, line and function name; the
            # library frames between the two are spliced out here.
            caught.__traceback__ = _user_traceback(caught, self.raw_func)
            raise caught from None

    def _sync_device(self, backend:str, device_obj:object)->None:
        """Wait until every launch this backend has queued has finished, callbacks included.

        The statements between two passes read shared slots on the host; the copies that
        fill them are stream callbacks, so a device-level sync is the one wait that is
        correct regardless of which stream each launch used.
        """
        if backend == "cuda":
            from ..cuda.driver import CUDA
            CUDA.cuCtxSynchronize()
        elif backend == "hip":
            from ..hip.driver import HIP
            HIP.hipDeviceSynchronize()
        elif backend == "metal":
            # Metal's launch waits on its own command buffer, so every launch that
            # returned has already completed; the queue is serial, which makes that
            # wait cover the launches before it too.
            device_obj.default_context.sync()

    def _run_host_region(self, statements:List[ast.stmt],
                         args:Tuple[object, ...], instance:object,
                         carried_slots:Dict[str, object], gap_slots:Dict[str, object],
                         globalns:Optional[Dict[str, object]]=None,
                         localns:Optional[Dict[str, object]]=None,
                         created_objects:Optional[Dict[str, object]]=None,
                         outer_state:Optional[OuterState]=None,
                         writable_outer:Optional[Dict[str, np.ndarray]]=None)->object:
        """Run one region of python statements on the host, and return a tail `return`.

        A region is the statements between two passes, or after the last one. A statement
        reads the launch arguments and whatever earlier statements and slots hold, exactly
        as python would. An array it touches is pulled back to the host first (the previous
        pass wrote it on the device) and marked dirty again afterwards, so the next launch
        pushes the changed values back. ``globalns``/``localns`` may pass a namespace in so
        consecutive regions share one scope, the way python's function body does.
        """
        if not statements:
            return None

        if globalns is None:
            globalns = dict(self.raw_func.__globals__)
        if localns is None:
            localns = {}
            for name, value in zip(_positional_argument_names(self.raw_func), args):
                localns[name] = value
            receiver:str = _receiver_name(self.raw_func)
            if receiver and instance is not None:
                localns[receiver] = instance
        # Closure cells the body reads (a factory returning a @kernel is the common shape)
        # and the outer-scope values the launch carries. A writable outer slot holds the
        # value the device may have updated, and a host write back into it travels to the
        # scope through the same buffer _write_back_outer copies from.
        for name, cell in zip(self.raw_func.__code__.co_freevars,
                              self.raw_func.__closure__ or ()):
            try:
                localns.setdefault(name, cell.cell_contents)
            except ValueError:
                continue                  # an empty cell: python would raise too
        for slot in (outer_state["writable"] if outer_state else []):
            localns[slot["name"]] = slot["buffer"][0]
        for name, buffer in (writable_outer or {}).items():
            localns[name] = buffer[0]
        # The slots the launch shares with the device: a carried scalar or a gap value that
        # an earlier phase updated is the current value, not the initial one. A created
        # array is the object itself -- indexing it would hand the statements its first row.
        for name, slot in list(carried_slots.items()) + list(gap_slots.items()):
            localns[name] = slot[0]
        # A created array is the object itself: unwrapping it would hand the statements its
        # first element (or row), which is not what python's name means.
        for name, value in (created_objects or {}).items():
            localns[name] = value

        touched:List[object] = []
        for statement in statements:
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and node.id in localns:
                    value:object = localns[node.id]
                    if isinstance(value, gnp):
                        value.to_host()
                        if value not in touched:
                            touched.append(value)
            if isinstance(statement, ast.Return):
                # A tail `return` is evaluated after every statement has run, not compiled
                # as part of the exec body: a bare `return` is illegal outside a function.
                continue
            if isinstance(statement, ast.AugAssign):
                # `count += 1` in a region updates the slot it names, the same way an
                # assignment does; the shared helper does not answer for this spelling.
                target_node:ast.expr = statement.target
                target:Optional[str] = (target_node.id
                                        if isinstance(target_node, ast.Name) else None)
            else:
                target:Optional[str] = _assigned_name_of(statement)
            # compile() needs a Module node, and the statement is one entry of the body.
            module:ast.Module = ast.Module(body=[statement], type_ignores=[])
            exec(compile(module, "<kernel gap>", "exec"), globalns, localns)  # noqa: S102 - the kernel source itself
            if target is not None and target in localns:
                value:object = localns[target]
                if isinstance(value, gnp) and value not in touched:
                    touched.append(value)
                elif isinstance(value, np.ndarray) and value.ndim > 0:
                    # A creation the gap statement built: the object itself is what the next
                    # pass receives, exactly as it does for a creation above the loops.
                    (created_objects if created_objects is not None else {})[target] = value
                elif not isinstance(value, gnp):
                    # `or` cannot pick the slot: bool(np.array([0])) is False, which made a
                    # zero-valued slot read as missing and silently skipped the write-back.
                    slot:Optional[np.ndarray] = (gap_slots.get(target)
                                                 if gap_slots.get(target) is not None
                                                 else carried_slots.get(target))
                    if slot is None:
                        slot = (writable_outer or {}).get(target)
                    if slot is not None:
                        slot[0] = int(value)
                        localns[target] = slot[0]
        for value in touched:
            value._set_host_dirty()
        for statement in statements:
            if isinstance(statement, ast.Return):
                if statement.value is None:
                    return None
                return eval(  # noqa: S307 - the kernel source itself
                    compile(ast.Expression(body=statement.value), "<kernel tail>", "eval"),
                    globalns, localns)
        return None

    def _invoke_inner(self, instance:object, args:Tuple[object, ...],
                      backend_sel:Optional[str], device_sel:Optional[str],
                      global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                      launch:str = "call")->object:
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
        device_obj:object
        device_key:str
        device_obj, device_key = _resolve_device(backend, device_sel)
        device_max_group, device_max_items = device_limits(device_obj)

        self_values:List[object] = []
        self_attrs:List[str] = []
        owner:Optional[type] = None
        self_struct_members:Dict[str, str] = {}
        self_struct_type:object = None
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
                value:object = getattr(instance, name)
                if callable(value) and not isinstance(value, np.ndarray):
                    raise SyntaxError(
                        f"self.{name} is a function, not a value: a function cannot be passed "
                        f"to a device (call it as self.{name}(...) instead)") from None
                self_values.append(value)
            # The scalar members travel as ONE structure the body reads through (the way a
            # device helper already takes `Scene self`); an ndarray member stays its own
            # argument, because the read/write analysis and the shape plumbing need it
            # separately. The record is one element long so `self.<attr> = ...` writes
            # through it, and the struct pointer is read-only when the body never assigns
            # a member.
            struct_attrs:List[str] = [name for name in self_attrs
                                      if not isinstance(getattr(instance, name), np.ndarray)]
            array_attrs:List[str] = [name for name in self_attrs
                                     if isinstance(getattr(instance, name), np.ndarray)]
            if struct_attrs:
                from .objects import _structure_for, member_type  # noqa: PLC0415 (import cycle)

                self_struct:type = _structure_for(
                    [(name, member_type(getattr(instance, name), name, self._type_map))
                     for name in struct_attrs], [])
                from .structs import register_name_hint  # noqa: PLC0415 (import cycle)

                register_name_hint(np.dtype(self_struct), owner)
                # Each attribute is a member of the same name: the mapping is the identity,
                # never a zip against the dtype's names. numpy sorts a record's field names
                # alphabetically, so zipping them against the attribute order paired
                # `image_height` with `du_in_world` and every member read the wrong field.
                self_struct_members = {name: name for name in struct_attrs}
                from . import structs as _structs  # noqa: PLC0415 (import cycle)

                self_struct_type = (PTR, _structs.register(np.dtype(self_struct)))
                # The value that travels is the ctypes structure itself: the launch packs
                # its bytes through the parsed layout, which handles vector members (whose
                # numpy record view uses a sub-array form that would misread the bytes).
                # A body that assigns `self.<attr> = ...` writes through the device buffer,
                # and the host copies the members home afterwards.
                holder:object = self_struct()
                for name in struct_attrs:
                    member_value:object = getattr(instance, name)
                    field_type:object = dict(self_struct._fields_).get(name)
                    if _is_complex_member_class(field_type) \
                            and not isinstance(member_value, field_type):
                        member_value = _complex_member_value(member_value, field_type)
                    setattr(holder, name, member_value)
                self_attrs = array_attrs
                self_types = [self_struct_type] + [
                    infer_ctype(getattr(instance, name), self._type_map)
                    for name in array_attrs]
                self_values = [holder] + [
                    getattr(instance, name) for name in array_attrs]

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
                                        self_value=instance,
                                        type_map=self._type_map)
        arg_types:List[DeviceType] = [infer_ctype(a, self._type_map) for a in args]
        if not self_struct_members:
            self_types = []
            for name, value in zip(self_attrs, self_values):
                try:
                    self_types.append(infer_ctype(value, self._type_map))
                except (SyntaxError, ValueError) as e:
                    # `ValueError` is the width refusal (a python int too wide for the
                    # mapping); both lose their own context here, so each is re-raised with
                    # the attribute that carries it.
                    raise type(e)(f"self.{name}: {e}") from None

        sig:str = type_sig(arg_types + self_types)
        # The mapping is part of the cache key, not only part of the type signature: two
        # kernels over the same argument values under different mappings produce `int` and
        # `long` parameters, and only one `_cache` entry may answer for a launch. The types
        # already differ (`sig` carries them), so this is belt-and-braces for the case where
        # a mapping is changed to a width whose type name another entry already used.
        cache_key:str = f"{backend}|{device_key}|{sig}|{self._type_map_key()}"
        entry = self._cache.get(cache_key)
        if entry is None:
            # Create the device objects for any Image before compiling. On the driver
            # this was developed against, loading a module that contains texture or
            # surface calls makes every later texture/surface creation fail, so an
            # image bound after compilation would never get its device storage.
            self._bind_images(full_values=args, backend=backend)
            entry = self._compile(args, arg_types, self_types, self_attrs, owner, sig,
                                  backend, device_obj, instance,
                      self_struct_members=self_struct_members,
                      self_struct_type=self_struct_type)
            self._cache[cache_key] = entry
        (kernel_obj, shape_sources, work_size_dims, local_sizes, uses_global_id,
         image_samplers, returns, outer_vars, launch_meta) = entry
        # One entry per magic loop, launched in order; a single-loop body has exactly one.
        active_kernels:List[object] = list(launch_meta.get("kernels") or [kernel_obj])
        # The extent arguments each phase kernel declares, in phase order.
        kernel_shapes:List[List[Tuple[int, int]]] = list(
            launch_meta.get("kernel_shapes") or [[] for _ in active_kernels])
        kernel_created:List[List[str]] = list(
            launch_meta.get("kernel_created") or [[] for _ in active_kernels])
        # Each phase's parameters in order, so the launch can align values to the signature
        # instead of assuming the shared prefix: a phase may declare a carried name the
        # first kernel does not, or skip one it does.
        phase_param_names:List[List[str]] = list(
            launch_meta.get("phase_param_names") or [[] for _ in active_kernels])
        # Whether each phase kernel declares the return claim/site words. A pass whose only
        # return runs in its host tail never claims, so its signature omits them and its
        # argument list has to omit them too (empty for a single-phase body, whose one
        # kernel is the main translator's own).
        phase_claims:List[bool] = list(launch_meta.get("phase_claims") or [])

        outer_writable:List[Dict[str, object]] = []
        carried_slots:Dict[str, object] = {}
        # The names `full` holds, in order: the caller's arguments, then the carried slots.
        # `build_phase_args` aligns each phase's signature against these names.
        # The `_self_<attr>` parameters sit in `full` right after the caller's arguments,
        # so the prefix names must include them or the aligned loop would look up an
        # attribute value in the wrong place.
        # When the receiver travels as a structure, it is the first entry of self_values
        # and the first device parameter: name it so the aligned loop finds the holder.
        receiver_name:List[str] = ([self._self_param]
                                   if self_struct_members else [])
        self_attr_names:List[str] = [f"_self_{name}" for name in self_attrs]
        # The launch is packed in the order the device signature declares: the receiver
        # first (it is a method's first parameter), then the caller's arguments, then the
        # ndarray attributes. `arg_types` and `self_types` already lead with the receiver
        # type, so the values have to lead with the receiver too.
        receiver_values:List[object] = (list(self_values[:1]) if self_struct_members else [])
        argument_values:List[object] = (list(self_values[1:]) if self_struct_members
                                     else list(self_values))
        prefix_names:List[str] = (receiver_name
                                  + list(_positional_argument_names(self.raw_func))[:len(args)]
                                  + self_attr_names
                                  + list(launch_meta.get("carried") or []))
        full:List[object] = [_kernel_scalar_argument(value, t)
                          for value, t in zip(receiver_values + list(args) + argument_values,
                                              arg_types + self_types)]
        # A carried scalar is one of the kernel's own parameters (declared before the shape
        # extents), so its slot joins full here rather than the trailing created values.
        for name in launch_meta.get("carried") or []:
            slot = np.zeros(1, dtype=np.int32)
            slot[0] = int(_host_scalar_values(
                self.raw_func, [name], tuple(args), instance).get(name, 0))
            full.append(slot)
            carried_slots[name] = slot
        own_count:int = len(full)

        # Outer-scope values are read before the launch arguments are built: a shape
        # source can name an outer array, and `full` is indexed by the device parameter
        # order (see _build_launch_args), so the outer values have to be part of it.
        outer_state:OuterState = self._bind_outer_values(outer_vars, backend)
        # Objects the kernel body creates are built here, on every call, exactly as the
        # interpreter would at that point in the body. They travel as trailing parameters.
        # A method's creation may read `self` (`np.zeros((self.height, ...))`), so the
        # instance is bound too.
        created:Dict[str, object] = _host_created_values(
            self.raw_func, tuple(args), instance,
            required_created=list(launch_meta.get("kernel_created_all") or []))[0]
        # A statement between two magic loops runs on the host, so its value is computed here
        # and travels to the pass that reads it. A scalar travels as a one-element slot,
        # exactly like a carried scalar; a creation the pass declares travels as the object
        # itself, exactly like any other created array.
        # One shared namespace per gap region: a later region reads what an earlier one
        # assigned, the way consecutive statements of a function body do.
        gap_namespaces:List[Tuple[Dict[str, object], Dict[str, object]]] = []
        for _ in (launch_meta.get("phase_gaps") or []):
            region_globals:Dict[str, object] = dict(self.raw_func.__globals__)
            region_locals:Dict[str, object] = {}
            for name, value in zip(_positional_argument_names(self.raw_func), args):
                region_locals[name] = value
            receiver:str = _receiver_name(self.raw_func)
            if receiver and instance is not None:
                region_locals[receiver] = instance
            gap_namespaces.append((region_globals, region_locals))
        gap_names:List[str] = list(launch_meta.get("gap_names") or [])
        created_all:List[str] = list(launch_meta.get("kernel_created_all") or [])
        gap_slots:Dict[str, object] = {}
        for name, value in _host_scalar_values(
                self.raw_func, [n for n in gap_names if n not in created_all],
                tuple(args), instance).items():
            slot = np.zeros(1, dtype=np.int32)
            slot[0] = int(value)
            gap_slots[name] = slot
        # The created objects the device signature declares: the aligned prefix picks the
        # real array from here rather than a scalar slot.
        created_objects:Dict[str, object] = {name: created[name] for name in created_all
                                          if name in created}

        outer_writable = list(outer_state["writable"])
        runtime_values:List[object] = list(outer_state["full"]) + _created_arguments(created)
        return_state:Dict[str, object] = (self._alloc_return_slots(returns)
                                      if returns else {})
        # Whether this launch's first kernel declares the return claim/site words. A
        # single-phase body has one kernel -- the main translator's -- and this is its own
        # answer; with several phases each entry of `phase_claims` decides instead. The
        # value is recorded at compile time because the translator is not reachable here.
        main_claims:bool = bool(launch_meta.get("main_claims", True))
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
                             extra_created:Optional[List[str]],
                             phase_index:int=0,
                             claims:Optional[bool]=None)->List[object]:
            """The launch arguments for one phase kernel.

            The shared prefix is the device parameter list up to `own_count`; the extents
            follow it because the translator appends them to each kernel's signature. The
            created objects differ per phase -- `mid = np.zeros(4)` between two loops is a
            parameter of the pass that reads it and of no other -- so `extra_created` names
            the ones this kernel expects.
            """
            # The prefix `full[:own_count]` holds the caller's arguments and the carried
            # slots by name; a phase's own signature may name a different subset (a gap
            # value only one pass reads, or a carried scalar this pass writes). Align the
            # values to the signature rather than assuming the prefix matches it.
            prefix_values:List[object] = list(full[:own_count])
            by_name:Dict[str, object] = dict(zip(prefix_names, prefix_values))
            # Only the names declared *before* the shape extents participate in the aligned
            # prefix: outer-scope values and host-created objects are declared after them and
            # the blocks below append those, so including them here would shift the extents.
            declared:List[str] = [name for name in (
                phase_param_names[phase_index]
                if phase_index < len(phase_param_names) else [])
                if name not in outer_vars and name not in created]
            if declared:
                # Align the values to this phase's own signature: a gap value only one pass
                # reads, or a carried scalar this pass writes, is not in the shared prefix.
                # Outer-scope names are excluded here -- the signature declares them last,
                # and the block below appends them in the host's own order.
                phase_args:List[object] = [by_name.get(name, gap_slots.get(name, 0))
                                        for name in declared]
            else:
                # No per-phase metadata (a gid-style body): the shared prefix *is* the
                # parameter list, in order.
                phase_args:List[object] = prefix_values
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
                # A created name this phase's own signature already declares (a gap creation
                # reads like `middle = np.zeros(4)` and the pass that uses it declares it)
                # has been passed in the aligned prefix above; adding it again here would
                # shift every later argument.
                if (extra_created is None or name in extra_created) and \
                        name not in declared:
                    phase_args.append(_kernel_scalar_argument(value, infer_ctype(value)))
            if return_state:
                phase_args.extend(return_state["buffers"])
                # A pass that never claims -- one whose only return runs in its host tail --
                # does not declare the words, so it must not receive them either. `claims`
                # is None only for the single-phase fast path, whose one kernel is the main
                # translator's: its own signature is what `main_claims` records.
                if claims if claims is not None else main_claims:
                    phase_args.append(return_state["claim"])
                    phase_args.append(return_state["site"])
            return phase_args

        # The read-back of a previous launch is a stream callback, and the caller may have
        # written the host arrays since. The upload below reads those arrays' memory, so it
        # must see the callbacks' result: wait once, before anything is uploaded.
        self._sync_device(backend, device_obj)
        launch_args:List[object] = build_phase_args(shape_sources, None)
        # Vector values travel in the layout the backend's ABI defines: OpenCL pads a
        # 3-component vector to four slots, CUDA/HIP take the host's three. The host keeps
        # the three-component form everywhere, so each argument is converted here and its
        # result copied back once the launch has finished.
        # Every entry pairs the value handed to the launch with the callable that copies it
        # back into the caller's own form once the launch has finished.
        unpackers:List[Tuple[object, Callable[[object], None]]] = []
        for index, argument in enumerate(launch_args):
            packed_value, unpack = pack(argument, backend)
            if packed_value is not argument:
                launch_args[index] = packed_value
            if unpack is not None:
                unpackers.append((packed_value, unpack))
        # The extents the magic loop itself iterates, in launch order (fastest axis first).
        # Derived whenever the loop states a domain, because an explicit bracket has to agree
        # with it axis by axis, not merely dimension by dimension.
        loop_extents:List[int] = []
        if work_size_dims:
            # A dim of -1 is an `@kernel(domain=...)` declaration: that array contributes
            # all of its dimensions, matching what np.ndindex(x.shape) would give. An entry
            # with no parameter index is a literal count, as in np.ndindex(2, 3).
            member_index:int = 0
            for pi, dim in work_size_dims:
                if pi is None:
                    loop_extents.append(dim)
                    continue
                if dim == -2:
                    # A host-side integer supplied as an argument, used as an extent:
                    # `np.ndindex(self.image_height, self.image_width)`.
                    loop_extents.append(int(full[pi]))
                    continue
                if dim == -3:
                    # The extent is a receiver-struct member: full[pi] is the ctypes
                    # structure the caller handed over, and the member holds the count.
                    member_name = launch_meta.get("work_size_members", [])[member_index]
                    member_index += 1
                    loop_extents.append(int(getattr(full[pi], member_name)))
                    continue
                shape:Tuple[int, ...] = tuple(int(s) for s in full[pi].shape)
                if dim >= 0:
                    if dim >= len(shape):
                        # The loop asks for an axis the array does not have. numpy raises
                        # this as an unpacking error from the `for` itself, so the launch
                        # says it in the same words instead of letting an index slip out.
                        raise ValueError(
                            f"{_unpack_count_message(dim + 1, len(shape))}: "
                            f"'{launch_meta['param_names'].get(pi, 'the array')}' has shape "
                            f"{shape}")
                    loop_extents.append(shape[dim])
                else:
                    loop_extents.extend(shape)
        if global_size is None and work_size_dims:
            # A dim of -1 is an `@kernel(domain=...)` declaration: that array contributes
            # all of its dimensions, matching what np.ndindex(x.shape) would give. An entry
            # with no parameter index is a literal count, as in np.ndindex(2, 3).
            extents_list:List[int] = list(loop_extents)
            # A domain of more axes than a device grid has (three) is folded into one flat
            # axis of `prod(shape)` work items, which is what the original auto-detection
            # did. Two consequences follow and both are honest: a magic loop recovers the
            # extra axes by decomposing that position over its own domain (`_emit_nd_head`),
            # while a gid-style body reads the flat position directly -- `get_global_id(0)`
            # is that index and the higher axes read 0, so a body that wants the original
            # axes spelled out should iterate with `np.ndindex` instead.
            extents:Tuple[int, ...] = tuple(extents_list)
            if len(extents) > 3:
                total:int = 1
                for extent in extents:
                    total *= int(extent)
                extents = (total,)
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
                    # numpy raises the unpacking error itself for this shape, so the device
                    # reports it in the same words: a body should not have to learn a second
                    # vocabulary for the same mistake. The array and its shape follow, since
                    # only the launch knows either.
                    raise ValueError(
                        f"{_unpack_count_message(loop_names, rank)}: "
                        f"'{launch_meta['param_names'].get(whole, 'the array')}' has shape "
                        f"{tuple(int(s) for s in full[whole].shape)}, and "
                        f"np.ndindex(....shape) iterates every dimension of it")
            # The loop has to name one index per dimension of its domain: `for j, i in
            # np.ndindex(image.shape)` over an RGB image would silently iterate only two of
            # the three dimensions, so the third axis never varies. Say so instead.
            if loop_names and loop_names != len(extents_list):
                source = ""
                if work_size_dims[0][0] is not None:
                    source = (f" (the array has shape "
                              f"{tuple(int(s) for s in full[work_size_dims[0][0]].shape)})")
                raise ValueError(
                    f"{_unpack_count_message(loop_names, len(extents_list))}: "
                    f"the loop's iteration domain has "
                    f"{len(extents_list)} dimension{'s' if len(extents_list) != 1 else ''}"
                    f"{source}. Give one name per dimension, or iterate an explicit subset, "
                    f"e.g. np.ndindex(image.shape[0], image.shape[1])")
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
                f"(`for i in range(x.shape[0]): ...`), name the array in the decorator "
                f"(`@kernel(domain=\"x\")`), or pass a bracket: "
                f"{self.__name__}[(total, local)](...)")
        if global_size is not None:
            # Whether the bracket agrees with the loop is a question about *coverage*, not
            # about rank: the kernel folds the launch's ids into one flat position and
            # decomposes it over the loop's own domain, so any launch shape works as long as
            # it supplies enough work items (see FunctionTranslator._emit_nd_head). A launch
            # enumerating fewer work items than the loop traverses stops short of part of the
            # domain and silently leaves it untouched, which is what this rejects. A larger
            # one is merely wasteful: the per-axis bounds guard skips the excess.
            #
            # The check only applies to a *magic* loop (`loop_names >= 1`): a gid-style body
            # states its domain with the bracket alone, so there is nothing to compare it with.
            loop_names:int = int(launch_meta.get("loop_names", 0))
            if loop_names and loop_extents:
                needed:int = 1
                for extent in loop_extents:
                    needed *= int(extent)
                supplied:int = 1
                for size in global_size:
                    supplied *= int(size)
                if supplied < needed:
                    raise ValueError(
                        f"{self.__name__}(): the work size bracket {tuple(global_size)} "
                        f"launches {supplied} work items, but its magic loop traverses "
                        f"{tuple(loop_extents)} ({needed} of them), so the rest of the domain "
                        f"would never be visited. Any launch shape is accepted as long as it "
                        f"covers {needed} work items (the loop decomposes a flat work-item "
                        f"position, so even {needed} alone works); a larger grid is fine, and "
                        f"omitting the bracket lets the loop derive the size itself")
            # The bracket arrived as `(blocks, threads)` and was turned into the total
            # work-item form here (see _parse_work_item), so the two axes agree in rank and
            # the total always divides by the group: the quotient *is* the block count. A
            # caller's own `local=` still has to be a size that splits the total, which is
            # the one divisibility question that remains -- and it is answered for that
            # argument, not for the bracket.
            if local_size is None:
                local_size = local_size_for_global(
                    global_size, max_work_group_size=device_max_group,
                    max_work_item_sizes=device_max_items)
            if len(local_size) != len(global_size):
                raise ValueError(
                    f"the block size {local_size} names {len(local_size)} "
                    f"dimension(s) but the launch has {len(global_size)}; both have to name "
                    f"the same number of axes")
            # Every backend's `Kernel.__getitem__` takes the same bracket the caller wrote
            # (`blocks, threads`), so what travels here is the block count and the group
            # width -- never the total. Handing a backend the total would make it multiply it
            # by the group width again: `(1, 256)` became 65536 work items that way. CUDA and
            # HIP take `(grid_dims, block_dims)` directly; OpenCL converts the pair to the
            # `(global_work_size, local_work_size)` its API wants (see utils.blocks_to_global).
            blocks = tuple(g // l for g, l in zip(global_size, local_size))
            phase_geometry = (blocks, local_size)
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
            if position:
                # The previous pass wrote the shared slots on the device; the read-back is
                # a stream callback that may not have run yet. The gap statements read those
                # slots on the host, so they must wait for the copies to land first.
                self._sync_device(backend, device_obj)
                regions:List[List[ast.stmt]] = list(launch_meta.get("phase_gaps") or [])
                if position - 1 < len(regions):
                    region_writable:Dict[str, np.ndarray] = {
                        entry["name"]: entry["buffer"] for entry in outer_writable}
                    self._run_host_region(regions[position - 1], args, instance,
                                          carried_slots, gap_slots,
                                          gap_namespaces[position - 1][0],
                                          gap_namespaces[position - 1][1],
                                          created_objects, outer_state,
                                          region_writable)
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
            # The fast path is only safe when the phase declares exactly the parameters the
            # first kernel does: a gap value (`factor = 10` between two loops) is one pass's
            # own parameter, and reusing the first kernel's argument list there would leave
            # every later argument one slot short.
            phase_names:List[str] = (phase_param_names[position]
                                     if position < len(phase_param_names) else [])
            first_names:List[str] = (phase_param_names[0]
                                     if phase_param_names else [])
            if (phase_shapes == shape_sources
                    and phase_created_names == list(created)
                    and phase_names == first_names):
                bound_kernel(*launch_args)
            else:
                bound_kernel(*build_phase_args(phase_shapes, phase_created_names, position,
                                               phase_claims[position]
                                               if position < len(phase_claims) else True))
        # The statements after the last loop are python: they run once, on the host, now that
        # every work item has finished -- the same moment python would run them.
        tail_writable:Dict[str, np.ndarray] = {
            entry["name"]: entry["buffer"] for entry in outer_state["writable"]} \
            if outer_state else {}
        tail_result:object = self._run_host_region(
            list(launch_meta.get("body_tail") or launch_meta.get("phase_tail") or []),
            args, instance,
            carried_slots, gap_slots, created_objects=created_objects,
            outer_state=outer_state, writable_outer=tail_writable)
        for packed_value, unpack in unpackers:
            unpack(packed_value)
        if return_state and int(return_state["claim"][0]):
            # A work item reached an in-loop return first: python's "first return wins".
            result:object = self._finish_returns(return_state, returns, full, created)
        elif tail_result is not None:
            result:object = tail_result
        else:
            result:object = (self._finish_returns(return_state, returns, full, created)
                          if return_state else None)
        self._write_back_outer(outer_state)
        if object_binding.written:
            object_binding.write_back()
        return result

    # ---------- outer-scope variables ----------

    def _bind_outer_values(self, outer_vars:Dict[str, object],
                           backend:str)->OuterState:
        """The launch arguments carrying this call's outer-scope values.

        A read-only outer variable is passed by value, refreshed on every call, so a
        changed scope value is seen immediately -- that is the whole point of
        parameterizing it instead of inlining it. A declared `nonlocal`/`global` scalar
        travels as a one-element buffer, because the device writes through the pointer;
        the buffer is copied back into the scope once the launch has completed (see
        _write_back_outer and _with_returns). An outer array is passed as itself, and the
        backend's read/write analysis reads it back when the kernel wrote it.
        """
        args:List[object] = []
        full:List[object] = []
        writable:List[Dict[str, object]] = []
        for name, outer in outer_vars.items():
            kind:str = outer["kind"]
            value:object = self._read_outer(name, kind, outer.get("scope"))
            if is_ptr(outer["type"]):
                if value is None:
                    raise ValueError(
                        f"outer array '{name}' is undefined at launch time")
                args.append(value)
                full.append(value)
                continue
            if not outer["writable"]:
                converted:object = _kernel_scalar_argument(value, outer["type"])
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
                    scope:Optional[Dict[str, object]] = None)->object:
        """The current value of an outer name, read the way python would resolve it.

        ``scope`` is the namespace the translator resolved the name in, which is the
        kernel's module for a name the body reads itself and the *helper's* module for one
        a device helper reads. Both are read live so a changed value is seen on the next
        call, which is why the namespace is carried rather than the value.
        """
        if kind == "closure":
            cells:Dict[str, object] = {}
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

    def _write_back_outer(self, outer_state:OuterState)->None:
        """Copy a declared outer scalar's final value back into its scope."""
        for entry in outer_state["writable"]:
            value:object = _outer_scope_value(entry["buffer"][0], entry["type"])
            self._store_outer(entry["name"], entry["kind"], value)

    def _store_outer(self, name:str, kind:str, value:object)->None:
        """Assign an outer name, through its closure cell or the module globals."""
        if kind == "closure":
            for cell_name, cell in zip(self.raw_func.__code__.co_freevars,
                                       self.raw_func.__closure__ or ()):
                if cell_name == name:
                    cell.cell_contents = value
                    return
            raise ValueError(f"outer variable '{name}' is not in the kernel's closure")
        self.raw_func.__globals__[name] = value

    def _alloc_return_slots(self, returns:ReturnSlots)->ReturnState:
        """One device buffer per value slot, plus the shared claim and site words.

        Every buffer is zero-initialised: the claim word then starts unclaimed, so a
        launch where no work item returned reads back as None rather than as garbage.
        """
        buffers:List[np.ndarray] = [np.zeros(1, dtype=dtype) for dtype in returns["slot_types"]]
        return {"buffers": buffers,
                "claim": np.zeros(1, dtype=np.int32),
                "site": np.zeros(1, dtype=np.int32)}

    def _finish_returns(self, return_state:ReturnState,
                        returns:ReturnSlots, full:List[object],
                        created:Optional[Dict[str, object]]=None)->object:
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
        layout:ReturnSiteLayout = returns["sites"][site]
        if not layout["positions"]:
            # The winning site was a bare `return` (or `return None`).
            return None
        values:List[object] = []
        for entry in layout["positions"]:
            if entry["kind"] == "array":
                name:str = entry["name"]
                if created is not None and name in created:
                    values.append(created[name])
                else:
                    values.append(full[returns["arg_indices"][name]])
            elif entry.get("reused"):
                # The value already lives in the argument the body returned, so the caller
                # gets it back from there -- no slot was allocated for it.
                # The parameter travels as a one-element buffer, so the value it holds is
                # what the caller passed -- and what the body may have updated. A numpy
                # record (`np.void`) is a parameter too, but indexing a record reads its
                # first *field*, so only a real 0-d/1-element array is unwrapped.
                shared_slot:object = full[returns["arg_indices"][entry["reused"]]]
                if isinstance(shared_slot, np.void):
                    values.append(shared_slot)
                else:
                    try:
                        values.append(shared_slot[0])
                    except (IndexError, TypeError):
                        values.append(shared_slot)
            else:
                dtype:object = returns["slot_types"][entry["slot"]]
                kind:str = returns["slot_kinds"][entry["slot"]]
                values.append(_return_value(return_state["buffers"][entry["slot"]][0], dtype, kind))
        return tuple(values) if layout["is_tuple"] else values[0]

    def _with_returns(self, future:object,
                      return_state:Dict[str, object],
                      returns:Optional[Dict[str, object]],
                      full:List[object],
                      outer_state:Optional[Dict[str, object]]=None,
                      created:Optional[Dict[str, object]]=None)->object:
        """Wrap an asynchronous launch so its future resolves to the returned value.

        The backend future resolves once every read-back has completed (see the kernel
        classes), so by the time this callback runs the slots hold the winner's values.
        A backend future may be bound to a loop other than the caller's (the python
        backend installs one when none is running), so an asyncio result is delivered
        through a fresh future on the caller's running loop when there is one, and is
        otherwise completed in place on whatever loop the backend future uses.
        """
        is_asyncio:bool = isinstance(future, asyncio.Future)
        deliver:object = concurrent.futures.Future()
        try:
            caller_loop:object = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None
        if is_asyncio:
            deliver = caller_loop.create_future() if caller_loop is not None else future

        def settle(*_args:object)->None:
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

    def _invoke_python(self, instance:object, args:Tuple[object, ...],
                       device_sel:Optional[str],
                       global_size:Optional[Tuple[int, ...]], local_size:Optional[Tuple[int, ...]],
                       launch:str)->object:
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
        runner:object = self._python_runner
        if runner is None:
            runner = PythonKernel(self.raw_func, self._is_method, self._self_param,
                                  domain=self._domain or "")
            self._python_runner = runner
        return runner.call(instance, args, global_size, local_size, launch=launch)

    # ---------- source generation ----------

    def _type_map_key(self)->str:
        """A stable spelling of this kernel's `int=`/`float=` mapping, for cache keys."""
        if not self._type_map:
            return ""
        return ",".join(f"{key}={self._type_map[key]}"
                        for key in sorted(self._type_map))

    def _source_signature(self)->str:
        """Python source signature: source changes -> regenerate (recompile when cached content differs).

        `@kernel(domain=...)` is deliberately **not** part of this hash, and that is safe only
        because the domain never reaches the generated code: it picks which array supplies the
        launch geometry, and geometry is a host-side, per-call fact. Two decorators over one
        body with different domains therefore share one source (they even compute one C name),
        and each instance keeps its own `_cache` entry so neither reuses the other's launch.
        If the domain ever starts to shape the device code, it has to be added here -- otherwise
        the two would share a source they no longer agree on.

        The `int=`/`float=` mapping **is** in this hash, because it does shape the device
        code: it decides which C type a python `int`/`float` parameter is declared as, so two
        mappings are two different translation units. Without it here, a kernel declaring
        `int=` would share the cache file (and the C name) of the default one, and the second
        launch would compile against a signature the caller did not ask for.
        """
        try:
            source:str = inspect.getsource(self.raw_func)
        except OSError:
            source = ""
        return short_hash(f"{source}\nbackend={self.backend}\ntypes={self._type_map_key()}")

    def _resolve_global(self, name:str)->object:
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

    def _c_kernel_name(self, sig:str, backend:str, declaration:str="")->str:
        """The kernel's C name, unique per module and per what it declares.

        Two modules can define a kernel with the same name and the same argument types --
        two test modules both declaring calls_a_method, say -- and one generated file
        #includes the other. Without the module in the hash the two kernels share one C
        name, and the source defines it twice.

        `declaration` is the rendered parameter list, when the caller already has one. The
        name has to cover it as well as `sig`: `sig` is built from the *argument types* the
        caller passed, while the generated signature also carries the extents derived from
        the body, the outer-scope values and the shared buffers. Without them in the hash a
        body whose derived signature changed kept its old name, and the file then held two
        declarations of that name. It is folded into the *same* hash rather than appended
        as a second one, so a kernel name stays `name_<hash>` (see docs/generated-source.md).
        """
        source_file:str = getattr(getattr(self.raw_func, "__code__", None),
                                  "co_filename", "") or ""
        module_mark:str = os.path.basename(source_file)
        # A method reads `Class_method`, exactly as a device helper does: the receiver is
        # the first parameter, so the class is part of what the kernel is. A plain function
        # keeps its own name.
        base:str = self.__name__
        qualname:str = getattr(self.raw_func, "__qualname__", "")
        if "." in qualname:
            owner_name:str = qualname.rsplit(".", 1)[0]
            if "<" not in owner_name:
                base = f"{owner_name}_{self.__name__}"
        payload:str = module_mark + "|" + sig + "_" + backend
        if declaration:
            payload += "|" + declaration
        return f"{base}_{short_hash(payload)}"

    def _bind_images(self, full_values:List[object], backend:str)->None:
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

    def _apply_declared_domain(self, translator:object)->None:
        """Record the array `@kernel(domain=...)` named as the iteration domain.

        The parameter has to exist and be an array: only a pointer carries the shape
        dimensions a launch is derived from, and a scalar has none to give. Naming a
        nonexistent parameter is the failure a string-valued decorator argument invites --
        renaming the parameter leaves the string behind -- so it is reported with the ones
        that do exist.

        A magic loop already fixes the domain, so a kernel carrying both would have two
        declarations of the same thing; that is checked here before translation, because the
        loop scan is what records `magic_loop_present`.
        """
        name:str = self._domain
        parameter_names:List[str] = _positional_argument_names(self.raw_func)
        if name not in parameter_names:
            available:str = ", ".join(parameter_names) or "none"
            raise SyntaxError(
                f"{self.__name__}(): @kernel(domain={name!r}) names no parameter of this "
                f"function (its parameters are: {available}). The domain is the array whose "
                f"shape declares the iteration domain, spelled as the parameter's own name")
        if _declares_a_magic_loop(self.raw_func):
            raise SyntaxError(
                f"{self.__name__}(): @kernel(domain={name!r}) declares the iteration domain, "
                f"but the body already declares one with a magic loop "
                f"(`for i in range(x.shape[0])` or `np.ndindex`); keep exactly one of them")
        index:Optional[int] = None
        for position, (parameter, _) in enumerate(translator.params):
            if parameter == name:
                index = position
                break
        if index is None or not is_ptr(translator.params[index][1]):
            raise SyntaxError(
                f"{self.__name__}(): @kernel(domain={name!r}) requires an array parameter; "
                f"'{name}' is not an array, so it has no shape to derive a work size from")
        translator.declared_work_arrays = [name]

    def _compile(self, args:List[object], arg_types:List[object], self_types:List[object],
                 self_attrs:List[str], owner:Optional[type], sig:str,
                 backend:str, device_obj:object, instance:object=None,
                 self_struct_members:Optional[Dict[str, str]]=None,
                 self_struct_type:object=None)->Tuple[object, List[str]]:
        if backend in ("cuda", "hip"):
            for t in list(arg_types) + list(self_types):
                base:str = (t[1] if isinstance(t, tuple) else t)
                if base == "half":
                    raise SyntaxError(
                        "fp16 (np.float16) arguments are not supported on the cuda backend: "
                        "NVRTC on this machine cannot resolve cuda_fp16.h; use np.float32 instead") from None

        c_name:str = self._c_kernel_name(sig, backend)
        device_cache:Dict[object, object] = {}
        # self_types leads with the receiver struct when the receiver travels as one, so
        # the attribute map pairs the array attributes with the types after it.
        self_attr_types:Dict[str, object] = dict(zip(
            self_attrs, self_types[1:] if self_struct_members else self_types))

        # hip accepts CUDA syntax (threadIdx/atomicAdd/make_float4 etc.), so codegen goes through cuda
        codegen_backend:str = ("cuda" if backend == "hip" else backend)
        # Statements above the magic loop that only build data are executed by the
        # interpreter, not translated: numpy decides the shape and dtype, and the resulting
        # object is what the kernel receives as a parameter (see
        # FunctionTranslator._host_created_assignment).
        creation_values, _ = _host_created_values(self.raw_func, args, instance)
        # The class each STRUCTURE argument was actually built from. One dtype can stand
        # for several classes, so the dtype alone cannot say whose method `p.total` is --
        # but the launch has the instance the caller handed over, and its type is the
        # answer. Keyed by parameter name, which is what the translator's expressions carry.
        # `structure_dtype_of` excludes the vector types, which are ctypes classes too.
        struct_arg_classes:Dict[str, type] = {}
        parameter_names:List[str] = _positional_argument_names(self.raw_func)
        for parameter_name, value in zip(parameter_names, args):
            # A union is a `ctypes` record as well but not a `Structure` subclass, so the
            # class is recognized through the shared predicate rather than `ctypes.Structure`.
            if not structs.is_ctypes_record(value):
                continue
            # structure_dtype_of excludes the vector types, which are ctypes classes too.
            if structs.structure_dtype_of(value) is None:
                continue
            struct_arg_classes[parameter_name] = type(value)
        translator:FunctionTranslator = FunctionTranslator(
            self.raw_func, list(arg_types) + list(self_types), codegen_backend,
            self._resolve_global, device_cache, is_kernel=True,
            self_param=self._self_param, self_attr_types=self_attr_types,
            self_owner=owner, launch_backend=backend,
            self_struct_members=self_struct_members,
            self_struct_type=self_struct_type,
            struct_arg_classes=struct_arg_classes,
            host_creation_values=creation_values,
            type_map=self._type_map)
        # `@kernel(domain="x")` names the parameter whose shape declares the iteration
        # domain. It is applied before `translate()` so the declaration is in place while the
        # magic loop is scanned -- a body cannot carry one of its own any more.
        if self._domain:
            self._apply_declared_domain(translator)
        translator.translate()
        # The name has to cover what the translator actually declares. `sig` is built from
        # the *argument types* the caller passed, which do not include the extent parameters
        # derived from the body, the outer-scope values or the shared buffers -- so a body
        # whose derived signature changed kept its old name, and the generated file ended up
        # holding two declarations of it ("more than one instance of overloaded function").
        # Re-deriving the name with the rendered declaration folds that into the same hash.
        c_name = self._c_kernel_name(sig, backend,
                                     declaration=";".join(translator.parameter_declarations()))
        # One kernel per magic loop: a body with several renders several, launched in order
        # so the arrays stay on the device between them.
        rendered:List[Tuple[str, str]] = translator.render_kernels(c_name)
        kernel_code:str = "\n\n".join(source for _, source in rendered)

        # The generated device code is split the way the python code is (see source_paths):
        # a kernel and every helper written in its own module go to that module's source
        # file, and a helper from another module goes to that module's file and is reached
        # with one `#include`. One file per python module means one compilation unit per
        # module, so a callee from another module still has to be reachable -- which is what
        # the include does (the preprocessor expands it before the compiler runs).
        kernel_source_file:str = getattr(self.raw_func.__code__, "co_filename", "") or ""
        # What the generated file records so a later call can tell whether it still
        # describes this module's text. mtime alone is not reliable (`git checkout` restores
        # the old stamp), so the digest is what actually decides.
        source_digest:str = source_paths.python_source_digest(kernel_source_file)

        def module_of(entry:Dict[str, object])->str:
            """The python file a device function came from, spelled as a cache path."""
            return source_paths.source_path(kernel_source_file, entry.get("source_file", ""),
                                            backend)

        # Device helper functions are emitted in dependency order: C requires a callee to be
        # declared before its caller, and the cache is filled depth-first (a nested helper
        # is finished *after* the helper that calls it). Emit leaves first.
        ordered_entries:List[Dict[str, object]] = _dependency_order(device_cache)
        own_entries:List[Dict[str, object]] = [
            entry for entry in ordered_entries
            if module_of(entry) == module_of({"source_file": kernel_source_file})]
        foreign_entries:Dict[str, List[Dict[str, object]]] = {}
        for entry in ordered_entries:
            path:str = module_of(entry)
            if path == module_of({"source_file": kernel_source_file}):
                continue
            foreign_entries.setdefault(path, []).append(entry)

        # A nested launch needs the callee's own `__global__` function in this same source:
        # the launch statement names a device function, and C requires it before its caller.
        # A built-in that a backend spells with a helper contributes one too, and it is a
        # device function the kernel calls, so it goes in the same place.
        device_prefix:List[str] = (list(translator.device_helpers)
                                   + list(translator.nested_helpers)
                                   + list(translator.nested_kernels))
        device_prefix.extend(entry["code"] for entry in own_entries)

        structure_dtypes:List[DeviceType] = list(translator.structure_dtypes())
        for entry in device_cache.values():
            for dtype in entry.get("structs", ()):
                if dtype not in structure_dtypes:
                    structure_dtypes.append(dtype)
        # Every structure this compilation reaches is declared by the kernel's own file. A
        # helper in another module names its types too, and that module's file is reached
        # from here, so the declaration has to be in scope before the include is read --
        # putting it in the helper's own file instead made a cycle of includes and, when two
        # modules shared one structure, the same declaration twice in one file.
        typedefs:str = (structs.render_structures(structure_dtypes, translator.backend)
                        if structure_dtypes else "")
        # A structure member whose type a fixed header declares needs that header even when
        # the body never operates on the value: `struct S { c64 z; };` names `c64`, so the
        # declaration has to be in scope. Walking the member dtypes here is what covers that
        # case -- the translator only records a width when an operation uses one.
        from . import complex as complex_support  # noqa: PLC0415 (import cycle)
        from . import quaternion as quaternion_support  # noqa: PLC0415 (import cycle)

        member_quaternion_needs:Set[str] = set()
        member_complex_needs:Set[str] = set()
        for dtype in structure_dtypes:
            record:np.dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
            for name in record.names or ():
                field_dtype:np.dtype = np.dtype(record.fields[name][0])
                if field_dtype.subdtype is not None:
                    field_dtype = field_dtype.subdtype[0]
                quat_need:Optional[str] = quaternion_support.needs(field_dtype)
                if quat_need is not None:
                    member_quaternion_needs.add(quat_need)
                complex_need:Optional[str] = complex_support.needs(field_dtype)
                if complex_need is not None:
                    member_complex_needs.add(complex_need)

        # Quaternion structures and their arithmetic both come from a fixed header, copied
        # into the cache's include folder beside the generated source. The source needs only
        # the enable macros for the widths in use, then one include -- no declaration of its
        # own, because the header spells the two type names in advance.
        quaternion_needs:Set[str] = set(translator.quaternion_needs)
        # A member declared by the fixed header needs it even when the body never
        # operates on the value (see the walk above).
        quaternion_needs.update(member_quaternion_needs)
        for entry in device_cache.values():
            quaternion_needs.update(entry.get("quaternion_needs") or ())
        include_lines:str = ""
        if quaternion_needs:
            from . import quaternion as quaternion_header

            include_lines = quaternion_header.include_line(
                tuple(sorted(quaternion_needs)), backend, self._cache_dir(backend))

        # Complex values work the same way, from their own fixed header: the two structures
        # and every operation on them are declared there, so a source that includes it says
        # nothing else about a complex number. A helper's blocks travel with its cache entry,
        # exactly as a helper's quaternion widths do.
        complex_needs:Set[str] = set(translator.complex_needs)
        complex_needs.update(member_complex_needs)
        for entry in device_cache.values():
            complex_needs.update(entry.get("complex_needs") or ())
        if complex_needs:
            from . import complex as complex_header

            complex_lines:str = complex_header.include_line(
                tuple(sorted(complex_needs)), backend, self._cache_dir(backend))
            include_lines = (f"{include_lines}\n{complex_lines}" if include_lines
                             else complex_lines)

        # A registered device-side library function may need headers the compilation does
        # not have yet. A header the user ships as a *path* is copied into the cache's
        # include folder and included by name; a header given as a full `#include` line
        # (a system header, say) is carried into the generated source verbatim.
        library_header_lines:List[str] = []
        if translator.device_library_headers:
            from .device_library import include_lines_for

            library_header_lines = include_lines_for(
                backend, translator.device_library_headers, self._cache_dir(backend))
        if library_header_lines:
            library_includes:str = "\n".join(library_header_lines)
            include_lines = (f"{include_lines}\n{library_includes}" if include_lines
                             else library_includes)

        # Every top-level definition is separated by one blank line: helper from helper,
        # helper from kernel. A trailing blank on each piece makes the join produce exactly
        # one, and the final trim drops the one after the last definition.
        pieces:List[str] = [piece.rstrip("\n") for piece in device_prefix]
        pieces.append(kernel_code.rstrip("\n"))

        own_path:str = module_of({"source_file": kernel_source_file})
        # The includes come first: a callee from another module is a declaration this file
        # needs before anything that calls it. The order is the order the callees were
        # emitted, so a foreign helper that itself includes another is already covered.
        # A structure declared by another module: that module's file holds the `struct`,
        # and this file includes it rather than repeating the declaration. A structure with
        # no python class behind it stays with the file that uses it.
        # The includes are ordered the same way the definitions are: a module whose code
        # calls another module's function has to be reached after that callee, because C
        # requires a function before its caller. The definitions are already in dependency
        # order, so the modules are taken in the order their first definition appears.
        ordered_foreign:List[str] = []
        for entry in ordered_entries:
            path:str = module_of(entry)
            if path in foreign_entries and path not in ordered_foreign:
                ordered_foreign.append(path)
        foreign_includes:List[str] = [
            source_paths.include_line(own_path, path) for path in ordered_foreign]
        # A generated file is a *header* plus a list of definitions. The header names every
        # structure the file's own code uses and includes every other module it reaches;
        # the definitions are appended one at a time, so a module with several kernels keeps
        # all of them in one file. The header comes first because C needs a type before the
        # function that names it, and a callee before its caller.
        own_header:List[str] = []
        if backend == "metal":
            # Metal's prelude: the `#include <metal_stdlib>` and the macros that give the
            # shared body its OpenCL index spellings (see kernel/metal_codegen.py). It
            # precedes the typedefs because the macros are the first thing the kernel body
            # needs -- and it must be first in the *file*, before any declaration.
            from .metal_codegen import prelude
            own_header.append(prelude())
        if include_lines:
            # The fixed headers come BEFORE the typedef block: a structure member can name
            # `quatf` or `c64`, and that declaration has to be in scope when the structure
            # is declared, not after it.
            own_header.append(include_lines)
        if typedefs:
            own_header.append(typedefs)
        if foreign_includes:
            own_header.append("\n".join(foreign_includes))

        written:Dict[str, str] = {}
        for path, entries in foreign_entries.items():
            # A foreign module's file declares no structures of its own. The kernel's file
            # carries the whole typedef block ahead of the includes that reach these files,
            # and each declaration is guarded by its structure name, so the type is declared
            # once no matter how many files are compiled from it. Repeating the block here
            # would put every declaration in every file, which is a cycle of includes and a
            # duplicate definition in the one file two modules share.
            foreign_header:List[str] = []
            if include_lines:
                foreign_header.append(include_lines)
            # A foreign module may itself call into a third module: every structure is
            # declared by the top-level file, but a *function* it calls has to be reached
            # from its own file, and only by the files that call it.
            callee_includes:List[str] = [
                source_paths.include_line(path, other) for other in ordered_foreign
                if other != path and other in _called_modules(path, foreign_entries)]
            if callee_includes:
                foreign_header.append("\n".join(callee_includes))
            written[path] = source_paths.append_source(
                path, foreign_header,
                [entry["code"] for entry in entries if entry["code"]],
                entries[0].get("source_file", "") or "",
                source_digest=source_digest)

        source_path:str = source_paths.append_source(
            own_path, own_header, [piece for piece in pieces if piece],
            kernel_source_file, source_digest=source_digest)
        # The whole file is one translation unit: a nested launch anywhere in it makes NVRTC
        # reject the source, so the *file* decides the build path, not the kernel being
        # compiled. A module with one nesting kernel and one ordinary kernel must be built
        # by nvcc even when the ordinary kernel is launched first, so the decision reads the
        # file that was actually written (it accumulates every kernel of the module).
        linked:bool = translator.nests_a_launch
        if not linked:
            try:
                with open(source_path, "r", encoding="utf-8") as file:
                    linked = "<<<" in file.read()
            except OSError:
                linked = False
        # The write above and the compile below are one critical section. A sibling kernel
        # of this same module may be appending its definition concurrently, and the compile
        # keys its cached binary on the file's *content*: a compile that started between two
        # appends would cache a binary of a half-populated file, and the kernel appended last
        # would find no entry point of its own in it (`KeyError` naming the kernel). Holding
        # the lock across both steps makes the file's content and the binary that describes
        # it change together.
        with source_paths.write_lock(source_path):
            kernels:List[object] = [
                self._backend_compile(source_path, name, backend, device_obj, linked=linked)
                for name, _ in rendered]
        # A phase kernel declares only the shapes it indexes itself, so its launch needs its
        # own trailing extent arguments rather than the ones the first kernel wants.
        extra_shapes:List[List[Tuple[int, int]]] = []
        # The created objects each phase declares: mid = np.zeros(4) between two loops is
        # a parameter of the pass that reads it and of no other.
        phase_created:List[List[str]] = []
        # The per-phase parameter metadata (empty for a single-phase body, which never
        # enters _translate_phase_kernels; the fallback below fills it from the translator).
        # A gid-style body has no phases and no magic loop: its launch arguments are the
        # shared prefix plus the extents, which the fast path already builds, and inventing
        # a per-phase list here would double-count the outer-scope parameters.
        is_magic_body:bool = bool(getattr(translator, "phases", None)) and \
            len(getattr(translator, "phases", [])) == 1 and \
            any(isinstance(stmt, ast.For) for stmt in translator.tree.body)
        phase_carried:List[List[str]] = []
        phase_param_names:List[List[str]] = []
        # Per phase, whether its kernel declares the return claim/site words (see
        # render_return_params): a pass whose only return runs in its host tail does not.
        phase_claims:List[bool] = []
        if translator.phase_translators:
            for phase in translator.phase_translators:
                phase_index:Dict[str, int] = {
                    name: position for position, (name, _) in enumerate(phase.params)}
                extra_shapes.append([(phase_index[array_name], dim)
                                     for array_name, dim in phase.shapes_used])
                phase_created.append(list(phase.host_values))
                phase_claims.append(bool(phase._claims_on_device))
        else:
            # One magic loop: the main translator *is* the only phase, and it binds the
            # created objects on itself.
            phase_index:Dict[str, int] = {
                name: position for position, (name, _) in enumerate(translator.params)}
            extra_shapes.append([(phase_index[array_name], dim)
                                 for array_name, dim in translator.shapes_used])
            phase_created.append(list(translator.host_values))
            # The one phase is the main translator: its parameters and carried slots are
            # the launch's, and the per-phase metadata would otherwise stay empty (the
            # launch then aligns against nothing and drops every argument after the first).
            if is_magic_body:
                phase_carried.append(list(translator.carried_scalars))
                phase_param_names.append(list(dict(translator.params)))
                phase_claims.append(bool(translator._claims_on_device))

        kernel:object = kernels[0]

        # Atomic targets: static analysis cannot recognize pointer-cast writes, so force them to writable
        from ..constants import KernelFlags
        for target_name in translator.atomic_targets:
            if target_name in kernel._args:
                kernel._args[target_name].kernel_flags = (kernel._args[target_name].kernel_flags | KernelFlags.Writed)

        # The return slots are written by the device and read back on the host, so they stay
        # fully bidirectional. `atomicCAS(&_ret_claim[0], 0, 1)` *reads* the claim word to
        # compare it against zero, and a slot that was never uploaded would compare against
        # whatever the device buffer happened to hold -- the claim would fire at random or
        # never, and a launch would return the wrong site's value. That is the same reason a
        # write-only argument is still uploaded, applied to a slot the library owns rather
        # than to the caller's data.
        for slot_name in (translator.RETURN_CLAIM, translator.RETURN_SITE):
            slot_info:object = kernel._args.get(slot_name)
            if slot_info is not None:
                slot_info.kernel_flags = (KernelFlags.Readed | KernelFlags.Writed)

        param_index:Dict[str, int] = {name: i for i, (name, _) in enumerate(translator.params)}
        # A multi-phase body is translated per phase, so the main translator indexes nothing;
        # the first phase's shapes are the ones this launch derives its work size from.
        shapes_owner:object = (translator.phase_translators[0]
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
            (param_index.get("self"), dim)
            if name.startswith("(self)->") else
            (shape_index[name], dim) if name else (None, dim)
            for name, dim in shapes_owner.work_size_dims]
        # An `@kernel(domain=...)` declaration: expand each named array into its own
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
        returns:Dict[str, object] = {
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
                 "carried": list(translator.carried_scalars),
                 "gap_names": list(translator.gap_names),
                 # The extra parameters each phase declares beyond the shared prefix: a gap
                 # value belongs to the pass that reads it (see _translate_phase_kernels).
                 "phase_carried": (phase_carried if phase_carried else
                                   [list(phase.carried_scalars)
                                    for phase in translator.phase_translators]),
                 # The statements between two passes: the host runs region i between
                 # launch i and launch i+1 (see _run_host_region).
                 "phase_gaps": getattr(translator, "phase_gaps", []),
                 # Each phase's parameters in order, so the launch aligns values to the
                 # signature (a phase may declare a carried name the first kernel does not).
                 "phase_param_names": (phase_param_names if phase_param_names else
                                       [list(dict(phase.params))
                                        for phase in translator.phase_translators]),
                 # Whether each phase's kernel declares the return claim/site words. A
                 # single-phase body has one kernel (the main translator's) and no entry
                 # here: the launch then treats it as claiming, which its signature says.
                 "phase_claims": (phase_claims if phase_claims else
                                  [bool(phase._claims_on_device)
                                   for phase in translator.phase_translators]),
                 # Whether the main kernel declares the claim/site words. The fast path
                 # launches that kernel with the argument list built once up front, so this
                 # has to describe the *first* kernel: a phase of a multi-loop body is one
                 # kernel of its own, while a single-loop body's only kernel is the main
                 # translator's (see render_return_params).
                 "main_claims": (bool(translator.phase_translators[0]._claims_on_device)
                                 if translator.phase_translators
                                 else bool(translator._claims_on_device)),
                 # The statements after the last magic loop, for a body split into passes:
                 # `body_tail` above belongs to a single-loop body, whose translation never
                 # enters _translate_phase_kernels. The last phase's own tail is the one the
                 # host runs after the final launch.
                 "phase_tail": (list(translator.phase_translators[-1].body_tail)
                                if translator.phase_translators else []),
                 # The receiver-struct members whose values are loop extents, in the
                 # order the -3 markers appear in work_size_dims (see _invoke's -3 branch).
                 "work_size_members": list(getattr(translator, "work_size_members", [])),
                 # The statements after the last magic loop of a single-phase body: the host
                 # runs them once, after the launch (see _run_host_region).
                 "body_tail": getattr(translator, "body_tail", []),
                 "kernels": kernels,
                 # The trailing extent arguments each phase kernel needs, by phase.
                 "kernel_shapes": extra_shapes,
                 "kernel_created": phase_created,
                 # Every created name any phase declares, so the host knows which objects
                 # the launch has to supply (a returned creation below the loop included).
                 "kernel_created_all": sorted({name
                                               for names in phase_created for name in names}),
                 "subscript_indices": dict(translator.subscript_indices),
                 # name -> the line of its widest subscript, and the offset that turns a
                 # parsed-copy line into the user's own: the launch blames that statement,
                 # not the loop, when a subscript names more indices than the rank.
                 "subscript_lines": dict(translator.subscript_lines),
                 "subscript_columns": dict(translator.subscript_columns),
                 # name -> the subscript as the user wrote it, so the frame stand-in can
                 # re-run that expression and let CPython draw its own caret.
                 "subscript_texts": dict(translator.subscript_texts),
                 "source_first_line": int(getattr(translator, "first_line", 1) or 1),
                 "loop_line": int(getattr(translator, "magic_loop_line", 0) or 0),
                 # How many index names the magic loop unpacks: the launch checks it against
                 # the number of dimensions its iteration domain actually has.
                 "loop_names": len(translator.work_item_indices),
                 # The array behind a `np.ndindex(arr.shape)` domain (index into the device
                 # parameter list, or -1), whose rank the loop must name one index per.
                 "whole_shape_index": param_index.get(translator.nd_whole_shape_param, -1)})

    def _cache_dir(self, backend:str)->str:
        """The pykernels folder this kernel's generated sources live in.

        It is the shared root the per-module files sit under, so the quaternion header can
        be placed there once and every module's source reach it.
        """
        source_file:str = getattr(self.raw_func.__code__, "co_filename", "") or ""
        return source_paths.cache_root(source_file, backend)

    def _backend_compile(self, source_path:str, c_name:str, backend:str, device_obj:object,
                         linked:bool=False):
        if backend == "opencl":
            from ..opencl.oop.program import Program
            program:object = Program(device_obj.platform.default_context, file_name=source_path)
            return program.kernels[c_name]

        if backend == "hip":
            from ..hip.oop.program import Program as HIPProgram
            from ..hip.oop.hip_compiler import HipCompiler
            device_obj.default_context.make_current()
            compiler = HipCompiler(source_path)
            program:object = HIPProgram(source_path, compiler=compiler)
            return program.kernels[c_name]

        if backend == "metal":
            from ..metal.oop.program import Program as MetalProgram
            program:object = MetalProgram(source_path, device=device_obj)
            return program.kernels[c_name]

        from ..cuda.oop.program import Program as CUProgram

        device_obj.default_context.make_current()
        if linked:
            # A kernel that launches another kernel needs *linked* device code: the child is
            # resolved through the device runtime, which NVRTC does not link in (and cuLink
            # on Windows will not read cudadevrt.lib). nvcc performs that link, so only such
            # a source pays for it; everything else stays on NVRTC.
            return _compile_linked_cuda(source_path, c_name, device_obj)

        program:object = CUProgram(source_path)
        return program.kernels[c_name]

    # ---------- launch ----------

    def _build_launch_args(self, args:List[object], arg_types:List[object], shape_sources:List[Tuple[int, int]])->List[object]:
        launch_args:List[object] = list(args)
        for param_index, dim in shape_sources:
            launch_args.append(int(args[param_index].shape[dim]))
        return launch_args

    def __repr__(self)->str:
        return f"PyKernel({self.__name__}, backend={self.backend})"


def _cuda_toolkit() -> Optional[pathlib.Path]:
    """The CUDA Toolkit directory, or None when nvcc cannot be found."""
    import shutil

    for variable in ("CUDA_PATH", "CUDA_HOME"):
        value:str = os.environ.get(variable, "")
        if value:
            candidate:pathlib.Path = pathlib.Path(value) / "bin" / (
                "nvcc.exe" if os.name == "nt" else "nvcc")
            if candidate.exists():
                return candidate
    # Linux installs ship /usr/local/cuda as a symlink to the versioned toolkit;
    # checking it keeps a toolkit without nvcc on PATH discoverable.
    if os.name != "nt":
        candidate:pathlib.Path = pathlib.Path("/usr/local/cuda/bin/nvcc")
        if candidate.exists():
            return candidate

    found:Optional[str] = shutil.which("nvcc")
    return pathlib.Path(found) if found else None


def _cuda_tool(nvcc:pathlib.Path, tool:str) -> Optional[pathlib.Path]:
    """Another executable shipped beside nvcc, or None when it is not installed."""
    import shutil

    tool_name:str = tool if os.name == "nt" else pathlib.Path(tool).stem
    candidate:pathlib.Path = nvcc.parent / tool_name
    if candidate.exists():
        return candidate

    found:Optional[str] = shutil.which(tool_name)
    return pathlib.Path(found) if found else None


class _LinkedImage:
    """A loaded module plus the metadata `Kernel` needs, for an nvcc-linked source.

    `Program` normally owns an NVRTC program and reads the kernel arguments from its
    parser. A linked build comes from nvcc instead, so only the two attributes `Kernel`
    actually touches are provided: the module handle and the per-kernel argument layout,
    which is parsed from the source the same way NVRTC's parser would.
    """

    def __init__(self, module:object, kernel_infos:Dict[str, object],
                 context:object, device:object)->None:
        self.id = module
        self.kernel_infos:Dict[str, object] = kernel_infos
        # `Kernel` reaches back into its program for these during a launch, so a linked
        # build has to answer them like an NVRTC one does.
        self.context:object = context
        self.device:object = device
        self.prefix:str = "nvcc"


def _compile_linked_cuda(source_path:str, c_name:str, device_obj:object)->object:
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
    executable:pathlib.Path = image_path.with_suffix(
        ".linked.exe" if os.name == "nt" else ".linked")
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
            "cuobjdump was not found beside nvcc; it is needed to take the linked device "
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
    kernel_infos:Dict[str, object] = dict(parser.kernel_infos)
    program:object = _LinkedImage(module, kernel_infos, device_obj.default_context, device_obj)
    kernel:object = Kernel(program, c_name)
    # `Kernel` starts with an empty argument layout and a `Program` fills it while fetching
    # its kernels; a linked build has no such program, so the parsed layout is handed over.
    kernel.args = kernel_infos[c_name].args
    kernel.type_checked = False
    return kernel


def _check_image_address(image:object, backend:str)->None:
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


def _image_sampler(image:"Image")->object:
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


def _outer_slot_dtype(t:DeviceType)->"np.dtype[Any]":
    """The numpy dtype of a one-element buffer carrying a declared outer scalar."""
    return np.dtype(_SCALAR_SLOT_DTYPES[t])


def _scalar_slot_value(value:object, t:DeviceType)->object:
    """An outer scalar as the element its buffer stores."""
    if t == "bool":
        return np.uint8(1 if value else 0)
    return value


def _outer_scope_value(raw:object, t:DeviceType)->object:
    """A declared outer scalar's buffer element as the python value to store back."""
    if t == "bool":
        return bool(int(raw))
    if t in ("char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"):
        return int(raw)
    if t in ("c64", "c128"):
        # A complex value comes back as a complex number, never as the record the buffer
        # holds: the caller assigned a `complex`, so that is what the scope gets again.
        return complex(raw)
    return float(raw)


_SCALAR_SLOT_DTYPES:Dict[str, object] = {
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
    # A complex value travels as a one-element complex buffer, which is the same bytes as
    # the fixed structure the device declares -- so a slot and its read-back agree.
    "c64": np.complex64,
    "c128": np.complex128,
}


def _check_shape_indices(kernel_name:str, full:List[object],
                         shape_sources:List[Tuple[int, int]],
                         work_size_dims:List[Tuple[Optional[int], int]],
                         launch_meta:Dict[str, object])->None:
    """Refuse a dimension the array does not have, in the caller's own terms.

    Two different mistakes reach this function, and they have to stay apart because python
    reports them differently:

    * a magic loop whose *iteration domain* names more axes than the array has
      (``for i, j in np.ndindex(arr.shape)`` over a 1-D array) is an unpacking error:
      ``not enough values to unpack (expected 2, got 1)``. Its rank is written in the
      source while the array's shape is known only from the value passed at call time, so
      it can only be found here.
    * a *subscript* that names more indices than the array has dimensions (``a[i, i]``
      over a 1-D array) is an ``IndexError``: ``too many indices for array: array is
      1-dimensional, but 2 were indexed``. A subscript naming fewer indices than the
      dimensions (``out[i, j]`` on a 3-D array) is legal python that selects a sub-array,
      which a work item cannot represent, so that one stays refused with its own wording.

    The two used to share one check driven by ``shape_sources``, which carries both the
    loop's extents and the dimensions a subscript asks for. A body indexing ``a[i, i]``
    therefore claimed the *loop* wanted two dimensions of ``a`` and reported an unpacking
    error about a loop that never iterated ``a`` at all.
    """
    param_names:Dict[int, str] = launch_meta["param_names"]
    name_to_index:Dict[str, int] = {name: index for index, name in param_names.items()}

    def shape_of(param_index:Optional[int])->Optional[Tuple[int, ...]]:
        if param_index is None:
            return None
        shape:Optional[Tuple[int, ...]] = getattr(full[param_index], "shape", None)
        return None if shape is None else tuple(int(s) for s in shape)

    # The loop's own iteration domain, and only that: every axis it names has to exist.
    first_line:int = int(launch_meta.get("source_first_line", 1) or 1)
    for param_index, dim in sorted({(pi, d) for pi, d in work_size_dims if pi is not None}):
        if dim < 0:
            continue                      # -1 is a whole-shape domain, checked below
        shape = shape_of(param_index)
        if shape is None:
            continue
        rank:int = len(shape)
        if dim < rank:
            continue
        name = param_names.get(param_index, f"argument {param_index}")
        error:BaseException = ValueError(
            f"{kernel_name}(): {_unpack_count_message(dim + 1, rank)}: the magic loop "
            f"iterates {dim + 1} axes but '{name}' has shape {shape}, so it has "
            f"{rank} dimension{'s' if rank != 1 else ''}")
        # The loop is the statement python blames, so point there.
        loop_line:int = int(launch_meta.get("loop_line", 0) or 0)
        if loop_line:
            error.lineno = first_line + loop_line - 1  # type: ignore[attr-defined]
        raise error from None

    # A subscript asks for *its* number of indices; that is what python counts.
    for name, count in sorted(launch_meta["subscript_indices"].items()):
        param_index:int = name_to_index.get(name, -1)
        if param_index < 0:
            continue                      # a local buffer or an image: no numpy value
        shape = shape_of(param_index)
        if shape is None:
            continue
        rank = len(shape)
        if count == rank:
            continue
        # The statement that subscripts the array is the one worth blaming, not the loop:
        # the parsed copy's line is offset by `source_first_line`, so the caret lands on
        # `a[i, i]` in the user's file, exactly where python would point.
        subscript_line:int = int(launch_meta.get("subscript_lines", {}).get(name, 0) or 0)
        first_line:int = int(launch_meta.get("source_first_line", 1) or 1)
        if count > rank:
            # numpy's own sentence and exception type, word for word: python raises
            # exactly this for `a[i, i]` on a 1-D array, so the device does too. Nothing
            # is appended -- the array is already named by the frame above, and an extra
            # clause made the two readings differ.
            error:BaseException = IndexError(
                f"too many indices for array: array is {rank}-dimensional, "
                f"but {count} were indexed")
            # What the frame stand-in needs to reproduce python's caret: the expression
            # that was too wide, and how wide the array really is.
            error.subscript_text = launch_meta.get(  # type: ignore[attr-defined]
                "subscript_texts", {}).get(name, "")
            error.array_rank = rank  # type: ignore[attr-defined]
        else:
            # Python raises nothing here -- `out[i, j]` on a 3-D array selects a sub-array --
            # so this is a device limit, not a mirror of the interpreter's. A ValueError
            # keeps it apart from the indexing errors above.
            error = ValueError(
                f"too few indices for '{name}': python would select a sub-array from its "
                f"shape {shape}, which a work item cannot represent. Index every dimension, "
                f"e.g. '{name}[i, j, k]' for a 3-D array")
        if subscript_line:
            # `_body_frame` reads these back to place a frame on that line, and the column
            # decides where CPython starts the caret -- on the subscript expression.
            error.lineno = first_line + subscript_line - 1  # type: ignore[attr-defined]
            error.column = int(  # type: ignore[attr-defined]
                launch_meta.get("subscript_columns", {}).get(name, 0) or 0)
        raise error from None


def _kernel_scalar_argument(value:object, t:DeviceType)->object:
    """A kernel argument in the type the device signature declares for it.

    Only `bool` needs converting: the translator passes it as a byte, because a real
    bool is rejected as a kernel parameter by OpenCL and has no CUDA binding entry. The
    conversion happens on the argument handed to the launch, not on the caller's object,
    so the values a return site hands back are still the caller's own objects.
    """
    if t == "bool":
        return np.uint8(1 if value else 0)
    return value


def _host_regions(func:object)->List[object]:
    """The statements a kernel body runs on the host, before and between its magic loops."""
    from .translator import _parse_kernel_source, _outside_loops, get_dedented_source

    try:
        tree:object = _parse_kernel_source(get_dedented_source(func))
    except SyntaxError:
        return []
    if not (tree.body and isinstance(tree.body[0], ast.ImportFrom)):
        pass
    body:List[object] = tree.body[0].body if tree.body else []
    return list(_outside_loops(body))


def _host_scalar_values(func:object, names:List[str], args:Optional[Tuple[object, ...]]=None,
                        instance:object=None)->Dict[str, object]:
    """Evaluate the initializers a carried scalar needs, on the host.

    A carried scalar (`total = 0` above the magic loop, `total = total + x[i]` inside it) is
    one location for the whole launch, so its starting value has to be a real python object
    -- the same one the interpreter would have made. The statements above the loop are run
    here in body order, sharing one namespace so a later initializer can read an earlier
    name, exactly as python would.
    """
    from .translator import _assigned_name_of

    values:Dict[str, object] = {}
    if not names:
        return values
    globalns:Dict[str, object] = dict(func.__globals__)
    localns:Dict[str, object] = {}
    from .decorator import _positional_argument_names
    for arg_name, value in zip(_positional_argument_names(func), args or ()):
        localns[arg_name] = value
    from .decorator import _receiver_name
    receiver:str = _receiver_name(func)
    if receiver and instance is not None:
        localns[receiver] = instance
    for name, cell in zip(func.__code__.co_freevars, func.__closure__ or ()):
        try:
            localns[name] = cell.cell_contents
        except ValueError:
            pass
    # Every statement that runs on the host, in body order: a gap statement is host code even
    # when the translator cannot lift it for a creation's sake, which is why the whole
    # host region is walked rather than only the statements a creation needs.
    statements = _host_regions(func)
    for statement in statements:
        target:Optional[str] = _assigned_name_of(statement)
        if target is None:
            continue
        try:
            localns[target] = eval(  # noqa: S307 - the kernel source itself
                compile(ast.Expression(body=statement.value), "<kernel scalar>", "eval"),
                globalns, localns)
        except Exception:
            continue
    for name in names:
        if name in localns:
            values[name] = localns[name]
    return values


def _reads_name(node:object, name:str)->bool:
    """Whether an expression reads the given plain name."""
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _host_created_values(func:object,
                         args:Optional[Tuple[object, ...]]=None,
                         instance:object=None,
                         required_created:Optional[List[str]]=None
                         )->Tuple[Dict[str, object], List[str]]:
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

    values:Dict[str, object] = {}
    names:List[str] = []
    try:
        assignments:List[object] = _creation_assignments(func)
    except SyntaxError:
        return values, names
    if not assignments:
        return values, names

    globalns:Dict[str, object] = dict(func.__globals__)
    localns:Dict[str, object] = {}
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
    wanted:set = set(required_created or [])
    for assignment in assignments:
        target:Optional[str] = _assigned_name_of(assignment)
        if target is None:
            continue
        if target not in wanted and _creation_target_of(assignment) is None:
            # A plain assignment is lifted only when a creation needs it (or a creation
            # it feeds was asked for by name).
            if not (target in localns and any(
                    _reads_name(other.value, target) for other in assignments)):
                continue
        try:
            value:object = eval(  # noqa: S307 - the kernel source itself
                compile(ast.Expression(body=assignment.value), "<kernel creation>", "eval"),
                globalns, localns)
        except Exception:
            continue  # the translator reports the real problem, with a source position
        localns[target] = value
        # Only a *creation* becomes a kernel parameter: its object has to reach the device.
        # A plain assignment lifted for the creation's sake stays in the namespace alone --
        # the work items compute it themselves, so shipping it would add an argument nothing
        # reads (and shift every later launch argument by one).
        if _creation_target_of(assignment) is None and target not in wanted:
            continue
        if target in wanted or (_creation_target_of(assignment) is not None
                                and _is_creation_expression(assignment.value, func)):
            if _creation_target_of(assignment) is None:
                # A plain assignment the caller asked for feeds a creation below the loop;
                # it stays out of the launch values, it only fills the namespace.
                localns.setdefault(target, value)
                continue
            values[target] = value
            names.append(target)
    return values, names


def _receiver_name(func:object)->str:
    """The name of a method's receiver parameter (`self` / `cls`), or "" for a function."""
    code:object = getattr(func, "__code__", None)
    if code is None or not code.co_varnames:
        return ""
    first:str = code.co_varnames[0]
    return first if first in ("self", "cls") else ""


def _declares_a_magic_loop(func:Callable)->bool:
    """True when the kernel body carries a magic loop, i.e. it fixes its own domain.

    The body scan the interpreter already performs answers this without translating: it
    records the loop's form and the array it iterates.
    """
    return analyse(func, _receiver_name(func)).magic_form is not None


def _positional_argument_names(func:object)->List[str]:
    """The python parameter names of a kernel, skipping the `self` parameter if present."""
    code:object = getattr(func, "__code__", None)
    if code is None:
        return []
    names:List[str] = list(code.co_varnames[:code.co_argcount])
    if names and names[0] in ("self", "cls"):
        return names[1:]
    return names


def _created_arguments(created:Dict[str, object])->List[object]:
    """The launch arguments for the objects a kernel body created, in declaration order."""
    return [_kernel_scalar_argument(value, infer_ctype(value)) for value in created.values()]


def _return_value(raw:object, dtype:object, kind:str)->object:
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
    vector:object = _VEC_RETURN_TYPES.get(str(np.dtype(dtype)))
    if vector is not None:
        return vector.from_buffer_copy(np.asarray(raw).tobytes())
    value:object = np.asarray(raw).reshape(())[()]
    if isinstance(dtype, np.dtype) and dtype.fields is not None:
        # A structure slot holds a record: the caller gets the record itself, and reads its
        # members exactly as the kernel body did (`result["i"]`).
        return value
    if kind == "c64" or kind == "c128":
        # A complex slot holds the complex layout, so the value is already a complex number.
        return complex(value)
    if kind == "bool":
        return bool(int(value))
    if kind in ("char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"):
        return int(value)
    return float(value)


_VEC_RETURN_TYPES:Dict[str, object] = {}


def _called_modules(path:str, foreign_entries:Dict[str, List[Dict[str, object]]]) -> set:
    """The other generated files one module's definitions call functions from.

    A module's own file holds its definitions; a definition that names another module's
    device function needs that file, and C requires the callee before its caller. The
    reference is matched textually against the callee's generated name, the way
    `_dependency_order` orders the definitions themselves.
    """
    own_code:str = "\n\n".join(entry["code"] for entry in foreign_entries.get(path, ())
                               if entry["code"])
    called:set = set()
    for other, entries in foreign_entries.items():
        if other == path:
            continue
        for entry in entries:
            name:str = entry.get("c_name") or ""
            if name and name in own_code:
                called.add(other)
                break
    return called


def _dependency_order(device_cache:Dict[object, object])->List[Dict[str, object]]:
    """Order cached device functions so each callee precedes its callers.

    The cache is populated depth-first, so a nested helper (e.g. `clamp01` called by
    `weight`) is completed after the helper that calls it. Emitting in insertion order
    would then reference a function C has not seen yet. Names are matched textually
    against each entry's generated code.
    """
    entries:List[Dict[str, object]] = [e for e in device_cache.values() if e.get("status") == "done"]
    if len(entries) <= 1:
        return entries

    # name -> entry, for the callee lookup
    by_name:Dict[str, Dict[str, object]] = {e["c_name"]: e for e in entries}
    ordered:List[Dict[str, object]] = []
    visiting:set = set()
    done:set = set()

    def visit(entry:Dict[str, object]) -> None:
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


@overload
def kernel(func:Callable[_P, Any], *, backend:Optional[str]=None,
           domain:Optional[str]=None,
           int:object=None, float:object=None)->PyKernel[_P]: ...
@overload
def kernel(func:None=None, *, backend:Optional[str]=None,
           domain:Optional[str]=None,
           int:object=None, float:object=None)->Callable[[Callable[_P, Any]], PyKernel[_P]]: ...
def kernel(func:Optional[Callable[..., Any]]=None, *, backend:Optional[str]=None,
           domain:Optional[str]=None, int:object=None, float:object=None):
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

    Both spellings are declared as overloads so the decorated name keeps the function's own
    parameters: ``@kernel`` and ``@kernel(backend=...)`` return one and the same
    ``PyKernel[...]``, and a checker reports a call's missing or extra argument against the
    author's ``def`` rather than against a ``*args``-only decorator.

    `int=` / `float=` choose the device width a python ``int`` / ``float`` travels as, because
    neither has a width of its own::

        @kernel(float=np.float64)          # python floats and python complex values are 64-bit
        def accurate(scale, out):
            ...

        @kernel(int=np.int64)              # a python int is a 64-bit counter here
        def count(step, out):
            ...

    The value is a type object -- a numpy dtype/scalar type (``np.int64``), a ctypes scalar
    class (``ctypes.c_int64``) -- or a device type name (``"long"``). Only the two python
    builtins are mapped: a numpy scalar or array already names its own width and keeps it
    (``np.uint32(x)`` is still ``uint``). A python ``complex`` has no width either, so it
    follows ``float=``. The choice is fixed when the decorator runs and is part of the
    kernel's identity, so ``set_type_map()`` does not change a kernel that is already
    declared.
    """
    type_map:Optional[Dict[str, object]] = None
    if int is not None or float is not None:
        type_map = {}
        if int is not None:
            type_map["int"] = int
        if float is not None:
            type_map["float"] = float

    def decorator(f:Callable[..., Any])->PyKernel[object]:
        if backend is not None:
            return PyKernel(f, backend, explicit_backend=True, domain=domain,
                            type_map=type_map)
        # No backend named: the call-time default decides, so this snapshot is only used
        # for reporting until set_default_device() (or a bracket) has a say.
        return PyKernel(f, _default_backend(), explicit_backend=False, domain=domain,
                        type_map=type_map)

    if func is not None:
        return decorator(func)
    return decorator
