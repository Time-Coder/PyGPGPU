from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..driver import HIP

if TYPE_CHECKING:
    from .device import Device
    from .buffer import Buffer


class Context:
    """HIP 的上下文即"当前设备"(hipSetDevice); make_current 切换当前设备。"""

    _current_context:"Context" = None

    def __init__(self, device:"Device"):
        self._device:"Device" = device
        self._buffers:Dict[int, List[Buffer]] = {}

    @staticmethod
    def current()->Optional["Context"]:
        return Context._current_context

    def make_current(self)->None:
        if Context._current_context is self:
            return
        HIP.hipSetDevice(self._device.id)
        Context._current_context = self

    def sync(self)->None:
        self.make_current()
        HIP.hipDeviceSynchronize()

    def get_buffer(self, size:int=0)->"Buffer":
        available_buffer = None
        for buffer in self._buffers.get(size, []):
            if not buffer.using:
                available_buffer = buffer
                break

        if available_buffer is None:
            from .buffer import Buffer
            available_buffer = Buffer(size)
            self._buffers.setdefault(size, []).append(available_buffer)

        available_buffer.use()
        return available_buffer

    def create_buffer(self, size:int=0, flags:Any=None)->"Buffer":
        """与 gnp ndarray._to_device 的接口对齐。"""
        from .buffer import Buffer
        return Buffer(size)

    @property
    def device(self)->"Device":
        return self._device
