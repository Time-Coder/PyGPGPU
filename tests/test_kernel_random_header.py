"""The random header: the file, the macros, and the backend spellings.

The header is a plain C file beside the quaternion and complex ones, and it carries the
same contract: a generated source that includes it says nothing else about a random
number, and the macros in front of the include are what select the backend's spelling.

Three facts are load-bearing here.

* **The 64-bit word is spelled per backend.** OpenCL C and MSL have `ulong`; CUDA C has
  no `ulong` at all, so the header's word is `unsigned long long` there. A header that
  spelled it one way would fail to compile on the other backends.
* **The double block is behind a macro.** MSL has no `double` at all, and an OpenCL
  device without `cl_khr_fp64` rejects a `double` variable however unused it is -- so
  the block is only compiled when a double draw was emitted.
* **The transcendental helpers are macros.** OpenCL C 1.2 has only the overloaded form
  (`log`, `cos`), so a header that spelled `logf` would not compile there.
"""
from __future__ import annotations

import pathlib

import pytest


def _header_module():
    from pygpgpu.kernel import rng

    return rng


def test_the_header_is_a_real_file_in_the_package():
    """The header ships with the package, or an installed copy looks for a file nothing has."""
    rng = _header_module()

    assert rng.HEADER_SOURCE == str(
        pathlib.Path(rng.__file__).parent / "include" / "random.h")
    source = pathlib.Path(rng.HEADER_SOURCE)
    assert source.is_file(), f"missing header: {source}"
    assert source.parent.name == "include"
    assert source.name == "random.h"
    pyproject = (pathlib.Path(rng.__file__).parents[2] / "pyproject.toml").read_text(
        encoding="utf-8")
    # The package data glob already covers the quaternion and complex headers; this pins
    # that it is the shared `kernel/include/*.h` one, so all three travel together.
    assert "kernel/include/*.h" in pyproject


def test_installing_copies_the_header_verbatim(tmp_path):
    rng = _header_module()

    installed = rng.install_header(str(tmp_path))
    with open(installed, "r", encoding="utf-8") as handle:
        assert handle.read() == rng.header_source()
    assert installed == str(tmp_path / "include" / rng.HEADER_NAME)


def test_the_include_line_carries_the_backend_spelling(tmp_path):
    """CUDA has no `ulong`, so the word is spelled per backend."""
    rng = _header_module()

    cuda = rng.include_line(("float",), "cuda", str(tmp_path))
    assert "#define PYG_RNG_QUAL __device__" in cuda
    assert "#define PYG_RNG_WORD unsigned long long" in cuda
    assert cuda.endswith('#include "include/random.h"')

    opencl = rng.include_line(("float",), "opencl", str(tmp_path))
    assert opencl.splitlines()[0] == "#define PYG_RNG_QUAL "
    assert "#define PYG_RNG_WORD ulong" in opencl

    metal = rng.include_line(("float",), "metal", str(tmp_path))
    assert metal.splitlines()[0] == "#define PYG_RNG_QUAL "
    assert "#define PYG_RNG_WORD ulong" in metal

    hip = rng.include_line(("float",), "hip", str(tmp_path))
    assert "#define PYG_RNG_QUAL __device__" in hip
    assert "#define PYG_RNG_WORD unsigned long long" in hip


def test_the_double_block_is_only_opened_when_a_double_draw_was_emitted(tmp_path):
    """A device without fp64 -- and MSL, which has no `double` at all -- rejects one."""
    rng = _header_module()

    float_only = rng.include_line(("float",), "opencl", str(tmp_path))
    assert "PYG_RNG_NEED_DOUBLE" not in float_only

    with_double = rng.include_line(("double",), "opencl", str(tmp_path))
    assert "#define PYG_RNG_NEED_DOUBLE" in with_double


def test_the_header_uses_the_plain_transcendental_spellings():
    """OpenCL C 1.2 has only the overloaded form, so a `logf` would not compile there."""
    rng = _header_module()

    text = rng.header_source()
    assert "#define PYG_RNG_LOG(x) log(x)" in text
    assert "#define PYG_RNG_COS(x) cos(x)" in text
    assert "#define PYG_RNG_SQRT(x) sqrt(x)" in text
    assert "logf(" not in text
    assert "cosf(" not in text


def test_the_header_declares_the_state_and_the_draws():
    rng = _header_module()

    text = rng.header_source()
    assert "typedef struct {" in text
    assert "} pyg_rng;" in text
    for name in ("pyg_rng_next", "pyg_rng_init", "pyg_rng_uniform", "pyg_rng_range",
                 "pyg_rng_normal", "pyg_rng_int", "pyg_rng_int_inclusive"):
        assert name in text, name


def test_every_function_is_static_so_nvrtc_does_not_warn():
    """NVRTC reports #177-D for an unreferenced `static` function, so the attribute says
    that the warning is expected -- the same contract the complex header has."""
    rng = _header_module()

    text = rng.header_source()
    assert "#define PYG_RNG_UNUSED __attribute__((unused))" in text
    # Every function carries the qualifier and the attribute.
    for line in text.splitlines():
        if line.startswith("PYG_RNG_QUAL static"):
            assert "PYG_RNG_UNUSED" in line, line


def test_the_functions_are_the_ones_the_translator_calls():
    """The header's names and the translator's names are one list, spelled twice."""
    from pygpgpu.kernel.translator import random_emit

    text = _header_module().header_source()
    assert "pyg_rng_uniform" in text
    assert "pyg_rng_range" in text
    assert "pyg_rng_normal" in text
    assert "pyg_rng_int" in text
    assert "pyg_rng_int_inclusive" in text
    assert random_emit._RandomEmitMixin.RANDOM_STATE == "_pyg_rng"
    assert random_emit._RandomEmitMixin.RANDOM_SEED == "_pyg_random_seed"


def test_the_seed_setting_is_a_32_bit_integer():
    rng = _header_module()

    rng.set_seed(1234)
    assert rng.seed() == 1234
    # The launch seeds walk a stride, so two launches differ while the sequence stays
    # reproducible from the one base.
    first = rng.launch_seed()
    second = rng.launch_seed()
    assert first != second
    rng.set_seed(1234)
    assert rng.launch_seed() == first
    rng.set_seed(None)
    assert rng.seed() is None


def test_the_seed_refuses_a_value_out_of_range():
    rng = _header_module()

    with pytest.raises(TypeError):
        rng.set_seed("seven")
    with pytest.raises(ValueError):
        rng.set_seed(-1)
    with pytest.raises(ValueError):
        rng.set_seed(2 ** 32)
    rng.set_seed(None)
