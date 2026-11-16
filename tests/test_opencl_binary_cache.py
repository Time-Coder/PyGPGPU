"""Bugs in the OpenCL program cache: the per-device binary table, and the cache lock key.

`binaries` reads one buffer per device from `clGetProgramInfo`. Each entry's bytes live at
*that device's* index, and the loop used `binaries[0]` for every one of them -- so a
multi-device context cached device 0's program under every device's key, and the next load
answered `clCreateProgramWithBinary` with bytes built for a device it was not built for.
"""
from __future__ import annotations


from pygpgpu.opencl.oop import program as program_module


class _FakeDevice:
    def __init__(self, name: str) -> None:
        self.name = name
        self.unique_key = name

    def __repr__(self) -> str:
        return f"Device({self.name})"


def test_each_device_reads_its_own_binary_index():
    """The loop must index `binaries` by the device, not always by 0.

    Two devices with *different* payloads make the mistake visible: with `binaries[0]` both
    entries came back as the first device's bytes.
    """
    payloads = [b"first-device-program", b"second-device-program"]
    devices = [_FakeDevice("gpu0"), _FakeDevice("gpu1")]

    # The property reads `self.binary_sizes`, `self.devices`, `self.n_devices` and the
    # driver calls; a stand-in object carries those and reproduces the read.
    class _Probe:
        _binaries: dict = {}

        @property
        def n_devices(self):
            return len(devices)

        @property
        def devices(self):
            return devices

        @property
        def binary_sizes(self):
            return {device: len(payloads[i]) for i, device in enumerate(devices)}

        @property
        def id(self):
            return 0

    import ctypes

    # `clGetProgramInfo` is stubbed to fill `binaries[i]` with payload i, which is what the
    # driver does: the array is one pointer per device, in device order.
    def fake_get_program_info(_program_id, _key, size, out_arr, _size_out):
        if out_arr is None:
            return
        for i, payload in enumerate(payloads):
            ctypes.memmove(out_arr[i], payload, len(payload))

    original = program_module.CL.clGetProgramInfo
    program_module.CL.clGetProgramInfo = fake_get_program_info
    try:
        from pygpgpu.opencl.oop.program import Program

        result = Program.binaries.fget(_Probe())
    finally:
        program_module.CL.clGetProgramInfo = original

    assert result[devices[0]] == payloads[0]
    assert result[devices[1]] == payloads[1], (
        "device 1 received device 0's bytes -- the loop used binaries[0]")
