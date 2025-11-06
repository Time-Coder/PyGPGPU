from pygpgpu.opencl.oop import Platforms, Context
from pygpgpu.opencl.runtime import CL


context = Platforms[0].device(0).create_context()
program = context.compile("test.cl")
print(program.binaries)