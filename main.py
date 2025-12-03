from pygpgpu.cuda import CUDA
from pygpgpu.cuda.oop import Device, Devices

CUDA.print_call = True

for device in Devices:
    print(device.default_context.api_version)
