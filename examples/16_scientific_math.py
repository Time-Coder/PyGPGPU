"""The scientific-computing vocabulary, and naming a device function that exists.

Run it with:  python examples/16_scientific_math.py
"""
import os

import numpy as np

from pygpgpu import device_function, erf, kernel, normal_cdf, reciprocal_hypot, tgamma

# A device function of your own, the way a CUDA header declares one. The header is copied
# into the cache and included in the generated source, so the symbol is in scope.
HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mylib.h")


@device_function(
    cuda="my_fast_rcp",
    opencl="native_recip",
    headers={"cuda": HEADER},
)
def my_fast_rcp(x: float) -> float:
    """The host's implementation; the device calls the symbol the registry names."""
    return 1.0 / x


@kernel
def evaluate(x: np.ndarray, out: np.ndarray):
    """The functions both backends have."""
    for i in range(out.shape[0]):
        out[i] = erf(x[i]) + tgamma(x[i] + 1.0)


@kernel
def evaluate_with_the_registered_function(x: np.ndarray, out: np.ndarray):
    """A device function that is not a builtin."""
    for i in range(out.shape[0]):
        out[i] = my_fast_rcp(x[i])


def main() -> None:
    values = np.array([0.25, 0.5, 1.0, 2.0], dtype=np.float32)

    print("== the functions both backends have ==")
    out = np.zeros(len(values), dtype=np.float32)
    evaluate(values, out)
    for value, result in zip(values, out):
        print(f"  erf({value}) + tgamma({value} + 1) = {result:.6f}")
    print(f"  (the host says {erf(values[1]) + tgamma(values[1] + 1.0):.6f} for the second)")

    print("== a device function registered by name ==")
    out = np.zeros(len(values), dtype=np.float32)
    evaluate_with_the_registered_function(values, out)
    print(f"  my_fast_rcp -> {np.round(out, 6).tolist()}")

    print("== the CUDA-only ones ==")
    print(f"  normal_cdf(0.0) on the host = {normal_cdf(0.0)}")
    print(f"  reciprocal_hypot(3, 4) on the host = {reciprocal_hypot(3.0, 4.0)}")
    print("  (fast_exp, exact_recip, erf_inverse and the rest run on CUDA only;")
    print("   on OpenCL the translator refuses them by name.)")


if __name__ == "__main__":
    main()
