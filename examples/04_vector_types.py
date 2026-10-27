"""04_vector_types.py -- vector types (float4/...) + the dual-mode function lib.

- 50 vector types (float4/int3/uchar2/...): construction, members
  (.x/.y/.z/.w or .r/.g/.b/.a), swizzles (e.g. .xyzz), arithmetic (scalar
  broadcast), comparisons, and componentwise math.
- To pass an array into a kernel as a vector argument, build it with the
  matching vector type's dtype (a float4 array -> `np.zeros(n, dtype=float4.dtype)`)
  and fill the elements with float4 instances.
- `pygpgpu.funcs` (a.k.a. vec_types.funcs) is a "dual-mode" library: it runs
  directly in host python (on vector instances or scalars), and inside an
  `@kernel` it translates to the corresponding device builtin.

Run:
    python examples/04_vector_types.py
"""
import numpy as np

from pygpgpu import kernel, funcs, float4


@kernel
def normalize_points(points: np.ndarray, out: np.ndarray):
    """Normalize each row's 4-component vector: funcs.normalize -> device builtin."""
    for i, v in enumerate(points):
        out[i] = funcs.normalize(v)


@kernel
def swizzle_points(points: np.ndarray, out: np.ndarray):
    """Swizzle demo: reorder components to (x,w,z,y), scale by a scalar, then
    add a float4 instance."""
    for i in range(out.shape[0]):
        p = points[i]
        out[i] = (p.xwzy * 2.0) + float4(1, 0, 0, 0)


def main():
    # --- Host-side float4 usage (works outside kernels too) ---
    a = float4(1, 2, 3, 4)
    b = float4(0, 1, 3, 0)
    print("a + b:", (a + b).xyzw)          # arithmetic
    print("a's .gr (`r/g/b/a` and `x/y/z/w` alias the same storage):", a.gr)
    print("dot(a, b):", funcs.dot(a, b))   # dual-mode lib runs on the host
    print("length(a):", funcs.length(a))

    # --- Vector arrays into a kernel: build a structured array with float4.dtype,
    #     fill elements with float4() ---
    n = 8
    pts = np.array([float4(float(i + 1), 0.0, 0.0, 0.0) for i in range(n)], dtype=float4.dtype)

    norm_out = np.empty(n, dtype=float4.dtype)
    normalize_points(pts, norm_out)

    # Kernel output is a structured array too: .tolist() gives one tuple per
    # element; rebuild float4 to check the lengths.
    lengths = [funcs.length(float4(*row)) for row in norm_out.tolist()]
    assert all(abs(l - 1.0) < 1e-4 for l in lengths), lengths
    print("normalize_points OK - all lengths are 1")

    sw_out = np.empty(n, dtype=float4.dtype)
    swizzle_points(pts, sw_out)
    first = float4(*sw_out.tolist()[0])     # p.xwzy reorder: for p=(i,0,0,0) it stays (i,0,0,0)
    print("swizzle_points[0] =", first.xyzw, "expected (3, 0, 0, 0) for p=(1,0,0,0)")


if __name__ == "__main__":
    main()