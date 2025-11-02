from pygpgpu.opencl.raii import Platforms
from pygpgpu.opencl.runtime import CL


for platform in Platforms:
    print(platform.extensions_with_version)
    for device in platform:
        print(device.extensions_with_version)
