from pygpgpu.opencl.raii import Platforms
from pygpgpu.opencl.runtime.cltypes import cl_name_version


for platform in Platforms:
    print(platform)
