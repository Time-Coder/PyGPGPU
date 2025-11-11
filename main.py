# from pygpgpu.opencl.oop import Platforms, Context
# from pygpgpu.opencl.runtime import CL, cl_kernel_arg_type_qualifier

# context = Platforms[0].devices[0].create_context()
# program = context.compile("test.cl")
# print(program.test_kernel)

# from pygpgpu.opencl import char2, short2, int2, long2, float2, int3, sizeof

# a = int3()
# print(sizeof(a))

def test(a, b, c, d):
    print(a, b, c, d)

test(a=1,d=3)