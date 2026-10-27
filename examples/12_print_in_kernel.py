"""12_print_in_kernel.py -- print() inside a @kernel function.

A kernel body can call the ordinary Python print(): the translator rewrites it into device
printf calls that produce the same line, so debugging a kernel needs no handwritten C and
no temporary output buffer.

The translation reproduces python's text:
- arguments are separated by one space, and the line ends with a newline;
- bools print as True/False, integers as plain decimals;
- floats print in python's shortest round-trip form (2.0, 0.1, 1e+20), which is what str()
  gives for the same float32;
- vectors print as their repr, float4(1.0, 2.0, 3.0, 4.0).

Two things worth knowing:
- every line is one printf call site (a bool argument branches it into one per True/False,
  a float into one per digit count), so all of a line's parts stay together;
- the order of the lines is the device's scheduling order, not the loop's, because work
  items run concurrently. `k["python"]` prints in loop order, so it is the reference for
  the text, not for the ordering.

print() is available on the OpenCL and CUDA backends. The HIP backend rejects it on purpose:
on the tested ROCm runtime, launching a kernel that carries these printf call sites makes
the driver drop the device, so debugging output is simply not worth that.

Run:
    python examples/12_print_in_kernel.py
"""
import numpy as np
from pygpgpu import kernel, float4


@kernel
def inspect(a: np.ndarray, out: np.ndarray):
    """Print every element with its type-dependent rendering, then copy it through."""
    for i in range(out.shape[0]):
        print("i =", i, "a[i] =", a[i], "positive =", a[i] > 0.0)
        out[i] = a[i]


@kernel
def inspect_vectors(vecs: np.ndarray, out: np.ndarray):
    """A vector argument prints like its python repr."""
    for i in range(out.shape[0]):
        print("vec", i, "=", vecs[i])
        out[i] = vecs[i].x


def main():
    values = np.array([0.1, 2.0, 1.0 / 3.0, -0.0, 1e20, 1500000.0], dtype=np.float32)
    out = np.zeros_like(values)

    print("== interpreter (loop order, reference text) ==")
    inspect["python"](values, out)

    print("== cuda ==")
    inspect["cuda"](values, out)

    print("== opencl ==")
    inspect["opencl"](values, out)
    assert np.array_equal(out, values)

    vectors = np.zeros(2, dtype=float4.dtype)
    vectors[0] = (0.1, 2.0, -0.0, 1.0 / 3.0)
    vectors[1] = (1.5, -2.5, 0.25, 1e20)
    vector_out = np.zeros(2, dtype=np.float32)
    print("== vectors ==")
    inspect_vectors["cuda"](vectors, vector_out)


if __name__ == "__main__":
    main()
