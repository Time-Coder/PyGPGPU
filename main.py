from pygpgpu.opencl.oop import Platforms, Context
from pygpgpu.opencl.runtime import CL, cl_kernel_arg_type_qualifier, CLInfo

from pygpgpu.opencl import short2

import numpy as np

# CL.print_call = True

program = Platforms[0].compile("test.cl")
a = np.zeros((100,), dtype=np.int32)
program.test_kernel(a)
print(a)
