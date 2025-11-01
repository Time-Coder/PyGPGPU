from pygpgpu.opencl.raii import Platforms
from pygpgpu.opencl.runtime import CL

# CL.print_call = True


for platform in Platforms:
    print(platform.icd_suffix)
    # for device in platform:
    #     print("    ", device.max_samplers)
