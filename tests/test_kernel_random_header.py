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

import numpy as np



def _header_module():
    from pygpgpu.kernel import random as random_support

    return random_support


def test_the_header_is_a_real_file_in_the_package():
    """The header ships with the package, or an installed copy looks for a file nothing has."""
    random_support = _header_module()

    assert random_support.HEADER_SOURCE == str(
        pathlib.Path(random_support.__file__).parent / "include" / "random.h")
    source = pathlib.Path(random_support.HEADER_SOURCE)
    assert source.is_file(), f"missing header: {source}"
    assert source.parent.name == "include"
    assert source.name == "random.h"
    pyproject = (pathlib.Path(random_support.__file__).parents[2] / "pyproject.toml").read_text(
        encoding="utf-8")
    # The package data glob already covers the quaternion and complex headers; this pins
    # that it is the shared `kernel/include/*.h` one, so all three travel together.
    assert "kernel/include/*.h" in pyproject


def test_installing_copies_the_header_verbatim(tmp_path):
    random_support = _header_module()

    installed = random_support.install_header(str(tmp_path))
    with open(installed, "r", encoding="utf-8") as handle:
        assert handle.read() == random_support.header_source()
    assert installed == str(tmp_path / "include" / random_support.HEADER_NAME)


def test_the_include_line_carries_the_backend_spelling(tmp_path):
    """CUDA has no `ulong`, so the word is spelled per backend."""
    random_support = _header_module()

    cuda = random_support.include_line(("float",), "cuda", str(tmp_path))
    assert "#define RANDOM_QUAL __device__" in cuda
    assert "#define RANDOM_WORD unsigned long long" in cuda
    assert cuda.endswith('#include "include/random.h"')

    opencl = random_support.include_line(("float",), "opencl", str(tmp_path))
    assert opencl.splitlines()[0] == "#define RANDOM_QUAL "
    assert "#define RANDOM_WORD ulong" in opencl

    metal = random_support.include_line(("float",), "metal", str(tmp_path))
    assert metal.splitlines()[0] == "#define RANDOM_QUAL "
    assert "#define RANDOM_WORD ulong" in metal

    hip = random_support.include_line(("float",), "hip", str(tmp_path))
    assert "#define RANDOM_QUAL __device__" in hip
    assert "#define RANDOM_WORD unsigned long long" in hip


def test_the_double_block_is_only_opened_when_a_double_draw_was_emitted(tmp_path):
    """A device without fp64 -- and MSL, which has no `double` at all -- rejects one."""
    random_support = _header_module()

    float_only = random_support.include_line(("float",), "opencl", str(tmp_path))
    assert "RANDOM_NEED_DOUBLE" not in float_only

    with_double = random_support.include_line(("double",), "opencl", str(tmp_path))
    assert "#define RANDOM_NEED_DOUBLE" in with_double


def test_the_header_uses_the_plain_transcendental_spellings():
    """OpenCL C 1.2 has only the overloaded form, so a `logf` would not compile there."""
    random_support = _header_module()

    text = random_support.header_source()
    assert "#define RANDOM_LOG(x) log(x)" in text
    assert "#define RANDOM_COS(x) cos(x)" in text
    assert "#define RANDOM_SQRT(x) sqrt(x)" in text
    assert "logf(" not in text
    assert "cosf(" not in text


def test_the_header_declares_the_state_and_the_draws():
    random_support = _header_module()

    text = random_support.header_source()
    assert "typedef struct {" in text
    assert "} random_state;" in text
    for name in ("random_next", "random_init", "random_uniform", "random_range",
                 "random_normal", "random_int", "random_int_inclusive"):
        assert name in text, name


def test_every_function_is_static_so_nvrtc_does_not_warn():
    """NVRTC reports #177-D for an unreferenced `static` function, so the attribute says
    that the warning is expected -- the same contract the complex header has."""
    random_support = _header_module()

    text = random_support.header_source()
    assert "#define RANDOM_UNUSED __attribute__((unused))" in text
    # Every function carries the qualifier and the attribute.
    for line in text.splitlines():
        if line.startswith("RANDOM_QUAL static"):
            assert "RANDOM_UNUSED" in line, line


def test_the_functions_are_the_ones_the_translator_calls():
    """The header's names and the translator's names are one list, spelled twice."""
    from pygpgpu.kernel.translator import random_emit

    text = _header_module().header_source()
    assert "random_uniform" in text
    assert "random_range" in text
    assert "random_normal" in text
    assert "random_int" in text
    assert "random_int_inclusive" in text
    assert random_emit._RandomEmitMixin.RANDOM_STATE == "_random_state"
    assert random_emit._RandomEmitMixin.RANDOM_SEED == "_random_seed"


def test_the_seed_comes_from_the_host_generator_state():
    """A host `random.seed()` before the first draw replays the device sequence."""
    import random as stdlib_random

    support = _header_module()
    state = stdlib_random.getstate()
    try:
        stdlib_random.seed(1234)
        support._reset_generator_seeds()
        first = support.launch_seed((support.SOURCE_RANDOM,))
        support._reset_generator_seeds()
        stdlib_random.seed(1234)
        assert support.launch_seed((support.SOURCE_RANDOM,)) == first
        stdlib_random.seed(1235)
        support._reset_generator_seeds()
        assert support.launch_seed((support.SOURCE_RANDOM,)) != first
    finally:
        stdlib_random.setstate(state)
        support._reset_generator_seeds()


def test_the_numpy_seed_comes_from_numpy():
    """`np.random.seed()` seeds numpy draws independently of the stdlib generator."""
    support = _header_module()
    state = np.random.get_state()
    try:
        np.random.seed(17)
        support._reset_generator_seeds()
        first = support.launch_seed((support.SOURCE_NUMPY,))
        support._reset_generator_seeds()
        np.random.seed(17)
        assert support.launch_seed((support.SOURCE_NUMPY,)) == first
        np.random.seed(18)
        support._reset_generator_seeds()
        assert support.launch_seed((support.SOURCE_NUMPY,)) != first
    finally:
        np.random.set_state(state)
        support._reset_generator_seeds()


def test_launches_walk_a_stride_and_sources_combine():
    support = _header_module()
    support._reset_generator_seeds()
    first = support.launch_seed((support.SOURCE_RANDOM,))
    second = support.launch_seed((support.SOURCE_RANDOM,))
    assert first != second
    support._reset_generator_seeds()
    assert support.launch_seed((support.SOURCE_RANDOM,)) == first
    support._reset_generator_seeds()
    both = support.launch_seed((support.SOURCE_RANDOM, support.SOURCE_NUMPY))
    assert 0 <= both <= 0xFFFFFFFF
    support._reset_generator_seeds()
