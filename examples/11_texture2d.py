"""11_texture2d.py -- sample a 2D image in a CUDA kernel with tex2D.

The image class is named as it appears in kernel code, and is constructed from
whatever the host already has -- a file path, a numpy array, or just a shape:

    texture = cudaTextureObject_t("photo.png")      # read with imageio
    texture = cudaTextureObject_t(numpy_image)      # (height, width[, channels])
    texture = cudaTextureObject_t(shape=(h, w))     # allocate, fill later

Pass it to a kernel like any other argument; the backend binds it to whichever
device and context the launch uses, so nothing here mentions either:

    kernel["cuda"](texture, out, width, height)

A kernel samples it with ``tex2D(tex, u, v)``, or ``tex2D<float>`` in C. With
``normalized=True`` (the default) a pixel centre maps to
``((x + 0.5) / width, (y + 0.5) / height)``.

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
from pygpgpu.cuda.driver import CUDA, cudaTextureObject_t

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

    # Build the images before anything else runs: binding needs a current CUDA
    # context, and constructing one here is what establishes it.
    texture_a = cudaTextureObject_t(ramp)
    texture_b = cudaTextureObject_t(trailing)
    print("images                  :", texture_a, "and", texture_b)

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
    # device call, so a Python kernel takes the image exactly like the .cu one.
    out_c = np.zeros(width * height, dtype=np.float32)
    sample_python["cuda"](texture_a, out_c, np.int32(width), np.int32(height))
    print("python @kernel, tex2D   :", bool(
        np.array_equal(out_c.reshape(height, width), ramp)))

    print("sample of row 0         :", out_a[0][:6])


if __name__ == "__main__":
    main()
