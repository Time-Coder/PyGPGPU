"""One Metal device, and the device list the kernel bracket selects from.

Metal's device model is flat: `MTLCopyAllDevices()` (macOS) returns every GPU,
and there are no platforms or sub-devices. Device 0 is what
`MTLCreateSystemDefaultDevice()` names, which is the integrated or discrete GPU
the system prefers.

The device answers the two limits the shared work-size code reads
(`max_work_group_size` / `max_work_item_sizes`) so `utils.device_limits` can
derive a launch exactly as it does for the other backends: Metal calls these the
threads per threadgroup and the maximum threads per axis of a threadgroup.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..driver.metal import MTL, MetaMetal, _wait_ready
from ..driver.objc import ObjC
from .metalobject import MTLObject


class Device(MTLObject):

    def __init__(self, device_id:int)->None:
        MTLObject.__init__(self, device_id)
        self._name:str = ""
        self._max_threadgroup:Optional[Tuple[int, int, int]] = None
        self._default_context:Any = None

    @property
    def name(self)->str:
        if not self._name:
            # -name returns an NSString (+0); `describe` reads its UTF-8 bytes, so
            # the bridge never has to deal with CFString.
            self._name = ObjC.describe(ObjC.send_ptr(self.id, "name"))
        return self._name

    @property
    def vendor(self)->str:
        """Metal devices are Apple's, and a selector like `metal:apple` should match.

        The other backends report a vendor string from the driver; Metal has no such
        query, so this answers the one fact that is always true of it. The shared
        `_device_match_keys` reads `vendor` with a `getattr` fallback, which is why
        this could be absent -- but then `metal:apple` would silently match nothing.
        """
        return "Apple"

    @property
    def max_threadgroup(self)->Tuple[int, int, int]:
        """The largest threadgroup, as (x, y, z).

        Metal reports this as an `MTLSize` struct returned by value from
        `maxThreadsPerThreadgroup`; a struct-returning objc method uses the
        stret calling convention on x86_64 and the ordinary one on arm64. The
        bridge asks for it through a small out-parameter form instead: the
        struct is fetched as raw bytes so neither ABI has to be guessed here.
        """
        if self._max_threadgroup is None:
            self._max_threadgroup = ObjC.struct_result(
                self.id, "maxThreadsPerThreadgroup", 3)
        return self._max_threadgroup

    @property
    def max_work_group_size(self)->int:
        """The product of the three threadgroup axes: Metal's real group budget."""
        return self.max_threadgroup[0] * self.max_threadgroup[1] * self.max_threadgroup[2]

    @property
    def max_work_item_sizes(self)->Tuple[int, int, int]:
        return self.max_threadgroup

    @property
    def total_memory(self)->int:
        """The recommended working-set size, which is what Metal asks callers to respect."""
        return ObjC.send_int(self.id, "recommendedMaxWorkingSetSize")

    @property
    def registry_id(self)->int:
        """The system-wide unique id of this GPU (`MTLDevice.registryID`)."""
        return int(ObjC.send_int(self.id, "registryID"))

    @property
    def unique_key(self)->str:
        """A name that distinguishes one device from another on the same machine.

        The name alone is not enough: a Mac with two identical GPUs reports the same
        `name` for both, and `unique_key` is the per-device key a compiled kernel is
        cached under -- two devices sharing one key would reuse each other's build.
        Every other backend folds an identity fact into the key for this reason
        (CUDA appends the compute capability, HIP the architecture, OpenCL the
        version and driver version).

        Metal's identity fact is `registryID`: it is unique per GPU on the system and
        stable for the life of the process. `MTLDevice.registryID` also exists on
        macOS, so this is the `MTLDevice` selector rather than `MTLDevice` vs
        `MTLGPUFamily` guessing.
        """
        from ...utils import sanitize_filename
        return sanitize_filename(f"{self.name} {self.registry_id}")

    def create_context(self)->Any:
        from .context import Context
        return Context(self)

    @property
    def default_context(self)->Any:
        if self._default_context is None:
            from .context import Context
            self._default_context = Context(self)
        return self._default_context


class MetaDevices(type):
    """The device list, enumerated once and cached."""

    __devices:List[Device] = []
    __map:dict = {}

    def __fetch(self)->List[Device]:
        if MetaDevices.__devices:
            return MetaDevices.__devices
        _wait_ready()
        # MTLCopyAllDevices is macOS only; on a machine without it (or without
        # Metal at all) the driver raises its own OSError, which the probe
        # treats as "unavailable" -- the same contract as the other backends.
        #
        # `_wait_ready` only waits: the driver records its failure in
        # `MetaMetal.last_init_error` rather than raising, so the first real use
        # is where it surfaces. This IS that first use for anything selecting a
        # Metal device, and the raw failure is a `FileNotFoundError` about a path
        # under `/System/Library` -- which says nothing about the backend being
        # unavailable, or that macOS is what it needs. Re-raising with that in
        # the message is what makes `k["metal"]` readable on a Windows machine.
        try:
            ids:List[int] = MTL.device_ids()
        except OSError as caught:
            reason:str = str(getattr(MetaMetal, "last_init_error", "") or caught)
            raise RuntimeError(
                "the Metal backend is unavailable on this machine: it needs macOS "
                f"with a real Apple GPU (Metal.framework could not be loaded: {reason})"
            ) from None
        for object_id in ids:
            MetaDevices.__devices.append(Device(object_id))
        return MetaDevices.__devices

    def __len__(self)->int:
        return len(self.__fetch())

    def __iter__(self):
        return iter(self.__fetch())

    def __getitem__(self, index:int)->Device:
        return self.__fetch()[index]

    def device(self, keyword:str)->Device:
        keyword_lower = keyword.lower()
        if keyword_lower in MetaDevices.__map:
            return MetaDevices.__map[keyword_lower]
        name = keyword_lower
        index = 0
        if ":" in keyword_lower:
            name, index_part = keyword_lower.rsplit(":", 1)
            index = int(index_part)
        devices = self.__fetch()
        if name.isdigit():
            device = devices[int(name)]
        else:
            matches = [d for d in devices if name in d.name.lower()]
            if not matches:
                available = ", ".join(d.name for d in devices)
                raise ValueError(f"Device {keyword} not found; available: {available}")
            device = matches[index]
        MetaDevices.__map[keyword_lower] = device
        return device


class Devices(metaclass=MetaDevices):
    pass
