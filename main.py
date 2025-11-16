from pygpgpu.opencl.oop import Platforms, Context
from pygpgpu.opencl import compile, int2, char2

import numpy as np

# CL.print_call = True

program = compile("test.cl", type_checked=False)
a = np.zeros((10, 10), dtype=np.int32)
program.test_kernel(a, (10.0, 10.0))
print(a)
