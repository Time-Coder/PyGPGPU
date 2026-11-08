"""A multi-pass body that computes with complex values in one of its passes.

The complex header is selected from the needs the translator records, and a multi-loop
body's device code lives on its phase translators: the main translator never visits a
loop, so its own set is empty. Reading only the main one left the generated source calling
`c128_mul`/`c128_add` without declaring them, and the compile failed.
"""
import numpy as np
from pygpgpu import kernel


@kernel
def complex_second_pass(seed: complex, out: np.ndarray):
    """Pass 2 iterates a complex value; pass 1 fills the output with its seed."""
    for i in range(out.shape[0]):
        out[i] = 0.0
    for i in range(out.shape[0]):
        z = seed
        for _ in range(3):
            z = z * z + seed
        out[i] = abs(z)


def test_a_complex_value_in_a_later_pass_compiles_and_runs(gpu_backend):
    out = np.zeros(2, dtype=np.float64)
    complex_second_pass[gpu_backend](np.complex128(0.3 + 0.5j), out)
    z = 0.3 + 0.5j
    for _ in range(3):
        z = z * z + (0.3 + 0.5j)
    np.testing.assert_allclose(out, np.full(2, abs(z)), rtol=1e-6, atol=1e-6)
