from pygpgpu.opencl.oop import Platforms, Context
from pygpgpu.opencl.runtime import CL

CL.print_call = True

for platform in Platforms:
    for device in platform:
        context = device.create_context()
        print(context, context.platform)
