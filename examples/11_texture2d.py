"""11_texture2d.py -- sample a 2D image in a CUDA kernel with tex2D.

An OpenCL kernel takes an ``image2d_t`` and a sampler; CUDA splits that into an
*array* (the storage the texture hardware reads) and a *texture object* (the array
plus its sampling rules). PyGPGPU folds both into one call:

    texture = Devices[0].default_context.texture(image)

``image`` is a numpy array shaped ``(height, width)`` or ``(height, width,
channels)``; its dtype and channel count carry over to the device image. Pass the
texture to a kernel like any other argument and sample it with ``tex2D(tex, u, v)``.
With ``normalized=True`` (the default) a pixel centre maps to
``((x + 0.5) / width, (y + 0.5) / height)``.

Keep the pieces separately when the image is needed for other work too:

    array = ctx.create_array(width, height, np.float32)
    array.set_data(image)
    texture = ctx.create_texture(array)

One ordering rule applies on the driver this was developed against (CUDA 13.4):
once a module containing ``tex2D`` has been loaded, no further texture object can be
created, so build every texture before importing such a module. Textures built
earlier keep sampling correctly, which is what this example does.

Run:
    python examples/11_texture2d.py
"""
import importlib
import os
import sys
import tempfile

import numpy as np

from pygpgpu import kernel, cuimport
from pygpgpu.kernel import tex2D
from pygpgpu.cuda.driver import CUDA
from pygpgpu.cuda.oop import Devices

CUDA.print_info = False

# A handwritten kernel: the texture analogue of the .cl examples.
SOURCE = """
typedef unsigned long long cudaTextureObject_t;

__global__ void sample_flat(cudaTextureObject_t tex, float *out, int w, int h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w * h) {
        int px = i % w;
        int py = i / w;
        float u = ((float)px + 0.5f) / (float)w;
        float v = ((float)py + 0.5f) / (float)h;
        out[i] = tex2D<float>(tex, u, v);
    }
}

__global__ void sample_2d(cudaTextureObject_t tex, float *out, int w, int h)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px < w && py < h) {
        float u = ((float)px + 0.5f) / (float)w;
        float v = ((float)py + 0.5f) / (float)h;
        out[py * w + px] = tex2D<float>(tex, u, v);
    }
}
"""


@kernel
def sample_python(tex, out, width, height):
    """The same sampling as ``sample_flat``, written as a Python kernel."""
    for y in range(height):
        for x in range(width):
            out[y * width + x] = tex2D(tex, (x + 0.5) / width, (y + 0.5) / height)


def main():
    width, height = 16, 8
    ramp = np.arange(width * height, dtype=np.float32).reshape(height, width)
    trailing = ramp * 2.0 + 5.0

    context = Devices[0].default_context

    # Build both textures before the module below is imported.
    texture_a = context.texture(ramp)
    texture_b = context.texture(trailing)

    print("array upload/read-back  :", bool(
        np.array_equal(texture_a.array.to_numpy(), ramp)))

    folder = tempfile.mkdtemp()
    with open(os.path.join(folder, "texture2d.cu"), "w") as handle:
        handle.write(SOURCE)
    sys.path.insert(0, folder)
    cuimport.install()
    # Imported through importlib because the module is generated at run time, so a
    # static `import texture2d` would not resolve for a type checker.
    kernels = importlib.import_module("texture2d")

    out_a = np.zeros((height, width), dtype=np.float32)
    kernels.sample_flat[(width * height, 32)](texture_a, out_a,
                                              np.int32(width), np.int32(height))
    print("1D launch, pixel coords :", bool(np.array_equal(out_a, ramp)))

    out_b = np.zeros((height, width), dtype=np.float32)
    kernels.sample_2d[((width, height), (16, 8))](texture_b, out_b,
                                                  np.int32(width), np.int32(height))
    print("2D grid, block/thread   :", bool(np.array_equal(out_b, trailing)))

    # The same sampling from a @kernel body: the translator turns tex2D into the
    # device call, so a Python kernel takes the handle exactly like the .cu one.
    out_c = np.zeros(width * height, dtype=np.float32)
    sample_python["cuda"](texture_a, out_c, np.int32(width), np.int32(height))
    print("python @kernel, tex2D   :", bool(
        np.array_equal(out_c.reshape(height, width), ramp)))

    print("sample of row 0         :", out_a[0][:6])


if __name__ == "__main__":
    main()
