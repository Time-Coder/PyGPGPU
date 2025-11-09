from pygpgpu.opencl.oop import Platforms, Context
from pygpgpu.opencl.runtime import CL, cl_kernel_arg_type_qualifier

context = Platforms[0].devices[0].create_context()
program = context.compile("test.cl")
print(program.test_kernel)