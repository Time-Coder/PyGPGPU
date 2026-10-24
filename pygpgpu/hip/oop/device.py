from __future__ import annotations

from ctypes import c_char, byref
from typing import Optional, TYPE_CHECKING

from ..driver import HIP, CUdevice

if TYPE_CHECKING:
    from .context import Context


class Device:

    _props_buffer_size:int = 8192

    def __init__(self, device_id:int):
        self._id:int = device_id

        # hipDeviceProp_t 布局随版本变化: name 固定在偏移 0(char[256]),
        # gcnArchName 位置不定 -> 整块读出后扫描 b"gfx" 提取架构
        buf = (c_char * Device._props_buffer_size)()
        HIP.hipGetDeviceProperties(byref(buf), self._id)
        raw:bytes = bytes(buf)
        self._name:str = raw[:256].split(b"\0")[0].decode("utf-8", errors="replace").strip()
        arch_pos:int = raw.find(b"gfx")
        self._arch:str = (raw[arch_pos:].split(b"\0")[0].decode() if arch_pos >= 0 else "")
        self._vendor:str = "AMD"

        self._default_context:Optional[Context] = None

    @property
    def id(self)->int:
        return self._id

    @property
    def name(self)->str:
        return self._name

    @property
    def arch(self)->str:
        """GCN 架构名, 如 gfx1036(编译目标)。"""
        return self._arch

    @property
    def vendor(self)->str:
        return self._vendor

    @property
    def unique_key(self)->str:
        from ...utils import sanitize_filename
        return sanitize_filename(f"{self.name} {self.arch}")

    @property
    def default_context(self)->"Context":
        if self._default_context is None:
            from .context import Context
            self._default_context = Context(self)

        return self._default_context

    def __eq__(self, other:"Device")->bool:
        return (self.__class__ == other.__class__ and self._id == other._id)

    def __hash__(self)->int:
        return self._id

    def __repr__(self)->str:
        return f"Device('{self.name}')"
