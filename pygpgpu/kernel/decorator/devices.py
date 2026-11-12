# Backend and device selection: the probe order, the selectors, the defaults.
from __future__ import annotations
import threading
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
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

# The parsed shape of one default-device configuration: each entry is one selector with
# the backend and the device parts it named, None where the selector left a part open (a
# bare "cuda" names only the backend, a bare "gfx1036" only the device).
DefaultDeviceEntries = List[Tuple[Optional[str], Optional[str]]]

# The selector list a bracket-less call walks, installed through ``config.default_device``.
# None means "use DEFAULT_BACKEND_ORDER".
_default_devices:Optional[DefaultDeviceEntries] = None
_default_device_lock = threading.Lock()


def _parse_default_devices(selectors:Sequence[str], setting:str)->DefaultDeviceEntries:
    """Parse and validate default-device selector strings -- the same strings a bracket takes.

    ``setting`` names the configuration in the refusals: the process-wide
    ``config.default_device`` and the per-kernel ``@kernel(default_device=...)`` accept the
    same strings, and each error names its own.
    """
    parsed:DefaultDeviceEntries = []
    for selector in selectors:
        if not isinstance(selector, str):
            raise TypeError(
                "%s takes selector strings like 'opencl:cpu', got %s"
                % (setting, type(selector).__name__))
        backend, device = _parse_selector(selector)
        if backend is None and device is None:
            raise ValueError("empty device selector")
        parsed.append((backend, device))
    return parsed


def _parse_default_device_value(value:object, setting:str)->Optional[DefaultDeviceEntries]:
    """One default-device assignment in either accepted shape, parsed.

    A selector string is a one-entry list, a list/tuple is taken in order, ``None`` means
    "not configured" (whatever is less specific then decides), and anything else is
    refused with the setting's own name in the message.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_default_devices([value], setting)
    if isinstance(value, (list, tuple)):
        return _parse_default_devices(list(value), setting)
    raise TypeError(
        "%s takes a selector string like 'cuda:0', a list of them, or None; got %s"
        % (setting, type(value).__name__))


def _install_parsed_default_devices(entries:Optional[DefaultDeviceEntries])->None:
    """Install what a kernel call without a bracket runs on, in order of preference.

    ``config.default_device = ["opencl:cpu", "cuda"]`` makes a plain ``k(...)`` try the CPU
    OpenCL device first and fall back to cuda when that is unusable. Entries are the same
    selector strings the brackets accept -- ``"backend"``, ``"backend:device"`` or a bare
    device name -- and each is tried until one resolves.

    ``None`` (or an empty list) restores the built-in order (``cuda -> hip -> metal ->
    opencl``).

    A kernel decorated with its own ``@kernel(default_device=...)`` walks that list
    instead; this setting decides what an unspecified call means for kernels that did not
    name one. The list is read at call time, so it applies to kernels that were already
    decorated.
    """
    global _default_devices
    with _default_device_lock:
        _default_devices = (entries or None)


def _selectors_of(entries:DefaultDeviceEntries)->List[str]:
    """Render parsed entries back to the selector strings they came from."""
    return [("%s:%s" % (backend, device)) if backend and device
            else (backend or device or "")
            for backend, device in entries]


def _default_device_selectors()->List[str]:
    """The selector list an unspecified kernel call currently walks."""
    with _default_device_lock:
        entries = _default_devices
    if entries is None:
        return list(DEFAULT_BACKEND_ORDER)
    return _selectors_of(entries)


def _entry_usable(backend:Optional[str], device:Optional[str])->Optional[Tuple[str, Optional[str]]]:
    """Resolve one default entry to a (backend, device) pair, or None if unusable.

    A backend-less entry (a bare device name) is looked up across the standard order, so
    ``config.default_device = ["gfx1036"]`` finds it whichever backend owns it.
    """
    if backend == "python":
        # The interpreter backend has no devices to enumerate, so the probe cannot vouch
        # for it -- but it runs wherever this package imports, so it is usable by
        # definition. Only a device part in the same selector ("python:0") is skipped,
        # the way a device that does not resolve is.
        return ("python", None) if device is None else None
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


def _walk_entries(entries:DefaultDeviceEntries)->Optional[Tuple[str, Optional[str]]]:
    """Walk parsed default entries in order; the first usable one answers."""
    for backend, device in entries:
        resolved:Optional[Tuple[str, Optional[str]]] = _entry_usable(backend, device)
        if resolved is not None:
            return resolved
    return None


def _default_target(kernel_entries:Optional[DefaultDeviceEntries]=None)->Tuple[str, Optional[str]]:
    """The backend (and device) a call without a bracket should use.

    ``kernel_entries`` is the kernel's own ``@kernel(default_device=...)`` list, and it
    replaces the walk entirely: a kernel that named one never consults
    ``config.default_device``. Without one the configured list is walked, and the built-in
    probe order is the last resort. The walk runs per call, because either list may change
    and devices may come and go.
    """
    with _default_device_lock:
        entries:Optional[DefaultDeviceEntries] = (
            kernel_entries if kernel_entries is not None else _default_devices)
    if entries is not None:
        resolved:Optional[Tuple[str, Optional[str]]] = _walk_entries(entries)
        if resolved is not None:
            return resolved
        if kernel_entries is not None:
            setting:str = "@kernel(default_device=...)"
            advice:str = ("remove the @kernel(default_device=...) argument to follow "
                          "config.default_device")
        else:
            setting = "config.default_device"
            advice = "assign None to config.default_device to restore the built-in order"
        raise RuntimeError(
            "none of the devices in %s is usable: %s. "
            "Check the names against list_devices(), or %s."
            % (setting, ", ".join(_selectors_of(entries)), advice))
    return _builtin_probe_backend(), None


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


def _builtin_probe_backend()->str:
    """The first backend in DEFAULT_BACKEND_ORDER that enumerates a device, probed once.

    The answer is cached in-process: probing imports the backend packages, and the order
    itself does not change at runtime.
    """
    global _probed_backend
    if _probed_backend is None:
        for backend in DEFAULT_BACKEND_ORDER:
            if _probe_backend(backend) is not None:
                _probed_backend = backend
                break
        else:
            _probed_backend = "opencl"
    return _probed_backend


def _snapshot_backend(kernel_entries:Optional[DefaultDeviceEntries]=None)->str:
    """The backend the decorator reports for a kernel that has not launched yet.

    The real backend/device choice happens per call (``_default_target``); this only fills
    ``PyKernel.backend`` until the first launch overwrites it, so it never raises -- an
    all-unusable list decorates fine and fails at the launch that needs it, where the
    error can name the call. A kernel that named entries walks its own list here, one that
    did not walks the configured global, and the built-in probe order answers when neither
    resolves.
    """
    with _default_device_lock:
        entries:Optional[DefaultDeviceEntries] = (
            kernel_entries if kernel_entries is not None else _default_devices)
    if entries is not None:
        resolved:Optional[Tuple[str, Optional[str]]] = _walk_entries(entries)
        if resolved is not None:
            return resolved[0]
    return _builtin_probe_backend()


def _default_backend()->str:
    """The backend an unspecified call uses, honouring ``config.default_device``."""
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
