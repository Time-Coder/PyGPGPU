"""11_texture2d.py -- sample a 2D image in a CUDA kernel with tex2D.

An OpenCL kernel takes an ``image2d_t`` and a sampler; CUDA splits that into an
*array* (the storage the texture hardware reads) and a *texture object* (the
array plus its sampling rules). PyGPGPU exposes both through the context:

    array = ctx.create_array(width, height, np.float32)
    array.set_data(host_image)
    texture = ctx.create_texture(array, normalized=True)

The texture is then passed to a kernel as an opaque handle and sampled with
``tex2D<float>(tex, u, v)``. With ``normalized=True`` a pixel centre maps to
``((x + 0.5) / width, (y + 0.5) / height)``, which is how CUDA sampling is usually
written.

Order matters on the driver this was developed against (CUDA 13.4): once a module
containing ``tex2D`` has been loaded into the process, every later
``cuTexObjectCreate`` fails with CUDA_ERROR_INVALID_VALUE, while a module without
``tex2D`` leaves it working. So the images and textures are built *before* the
kernel module is imported, which this example does. ``Context.create_texture``
also retries a refusal by rebuilding the image, which covers the separate
cold-start refusals but cannot recover after a tex2D module has loaded.

Run:
    python examples/11_texture2d.py
"""
import importlib
import os
import sys
import tempfile

import numpy as np

from pygpgpu import kernel
from pygpgpu.kernel import tex2D
from pygpgpu import cuimport
from pygpgpu.cuda.driver import CUDA
from pygpgpu.cuda.oop.context import Context

CUDA.print_info = False

# A handwritten kernel: the texture analogue of the .cl examples. The texture is
# created with normalized coordinates, so each pixel centre maps to
# ((x + 0.5) / w, (y + 0.5) / h); with point sampling that returns the pixel the
# thread is responsible for.
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
def warm(x, out, a):
    for i in range(out.shape[0]):
        out[i] = x[i] * a


@kernel
def sample_python(tex, out, width, height):
    """The same sampling as ``sample_flat``, written as a Python kernel."""
    for y in range(height):
        for x in range(width):
            out[y * width + x] = tex2D(tex, (x + 0.5) / width, (y + 0.5) / height)


def main():
    # A launch first, so the CUDA context exists.
    warmup = np.random.rand(1024).astype(np.float32)
    warm["cuda"](warmup, np.zeros_like(warmup), 2.0)

    context = Context.current()
    width, height = 16, 8

    ramp = np.arange(width * height, dtype=np.float32).reshape(height, width)
    scaled = ramp * 2.0 + 5.0

    array_a = context.create_array(width, height, np.float32)
    array_a.set_data(ramp)
    print("array upload/read-back  :", bool(np.array_equal(array_a.to_numpy(), ramp)))

    try:
        texture_a = context.create_texture(array_a, normalized=True)
    except BaseException as error:
        print("this driver/interpreter combination refuses cuTexObjectCreate")
        print("(%s); the array APIs above still work" % type(error).__name__)
        return

    array_b = context.create_array(width, height, np.float32)
    array_b.set_data(scaled)
    texture_b = context.create_texture(array_b, normalized=True)

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
    print("2D grid, block/thread   :", bool(np.array_equal(out_b, scaled)))

    # The same sampling from a @kernel body: the translator turns tex2D into the
    # device call, so a Python kernel takes the handle exactly like the .cu one.
    out_c = np.zeros(width * height, dtype=np.float32)
    sample_python["cuda"](texture_a, out_c, np.int32(width), np.int32(height))
    print("python @kernel, tex2D   :", bool(
        np.array_equal(out_c.reshape(height, width), ramp)))

    print("sample of row 0         :", out_a[0][:6])


if __name__ == "__main__":
    main()



