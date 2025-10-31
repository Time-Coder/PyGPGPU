from pygpgpu.opencl.raii import Platforms


for platform in Platforms:
    print(platform, platform.n_devices)
    for i in range(platform.n_devices):
        print("   ", platform.device(i).type)