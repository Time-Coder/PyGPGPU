# Backend and device selection: the probe order, the selectors, the defaults.
from __future__ import annotations
import threading
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

from ...utils import blocks_to_global



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
        from ...opencl.oop.platforms import Platforms
        return [d for platform in Platforms for d in platform.devices]
    if backend == "hip":
        from ...hip.oop.devices import Devices
        return list(Devices)
    if backend == "metal":
        from ...metal.oop.devices import Devices
        return list(Devices)
    if backend == "cuda":
        from ...cuda.oop.devices import Devices
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


# Vendor spellings people reach for, mapped to a substring of the vendor string the
# drivers actually report. "amd" is the common case: the OpenCL vendor string is
# "Advanced Micro Devices, Inc.", which does not contain the abbreviation, so a plain
# substring match for "amd" would never find an AMD device.

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
