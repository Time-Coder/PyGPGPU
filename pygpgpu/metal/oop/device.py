"""The Metal `Device`, under the module name every other backend uses.

Every backend publishes its device class as `oop.device.Device` (OpenCL, CUDA and HIP
all define it there), and its device *list* as `oop.devices.Devices`. Metal keeps both
in `devices.py`, because the list's metaclass needs the class and the two are a few
dozen lines apart -- which is fine to read but breaks a caller that walks the backends
generically (`for backend in (...): import pygpgpu.{backend}.oop.device`).

This module is the shared name, re-exporting rather than redefining: one `Device`
class, reachable two ways. Nothing imports it internally; it exists for the cross-
backend surface.
"""
from __future__ import annotations

from .devices import Device

__all__ = ["Device"]
