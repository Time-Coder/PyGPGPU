"""The header tests, split out of `test_kernel_complex.py`.

The subject is header.
"""
from __future__ import annotations


import numpy as np


def test_the_header_is_a_real_file_in_the_package():
    """The header ships with the package, or an installed copy looks for a file nothing has."""
    from pygpgpu.kernel import complex as complex_header

    assert complex_header.HEADER_SOURCE == str(
        __import__("pathlib").Path(complex_header.__file__).parent / "include" / "complex.h")

    import pathlib

    source = pathlib.Path(complex_header.HEADER_SOURCE)
    assert source.is_file(), f"missing header: {source}"
    assert source.parent.name == "include"
    assert source.name == "complex.h"
    pyproject = (pathlib.Path(complex_header.__file__).parents[2] / "pyproject.toml").read_text(
        encoding="utf-8")
    # The package data glob already covers the quaternion header; this pins that it is the
    # shared `kernel/include/*.h` one, so both headers travel together.
    assert "kernel/include/*.h" in pyproject


def test_the_header_declares_both_structures_with_numpys_layout():
    """The member names are numpy's own (`re`/`im`), which is what makes an array bind."""
    from pygpgpu.kernel import complex as complex_header

    text = complex_header.header_source()
    assert "typedef struct { float re; float im; } c64;" in text
    assert "typedef struct { double re; double im; } c128;" in text
    # OpenCL C has no `sqrtf`, only the overloaded form, so the header must not name one.
    assert "sqrtf(" not in text


def test_installing_copies_the_header_verbatim(tmp_path):
    from pygpgpu.kernel import complex as complex_header

    installed = complex_header.install_header(str(tmp_path))
    with open(installed, "r", encoding="utf-8") as handle:
        assert handle.read() == complex_header.header_source()
    assert installed == str(tmp_path / "include" / complex_header.HEADER_NAME)


def test_the_include_line_enables_only_the_widths_in_use(tmp_path):
    from pygpgpu.kernel import complex as complex_header

    float_only = complex_header.include_line(("COMPLEX_NEED_FLOAT",), "cuda", str(tmp_path))
    assert float_only == ('#define COMPLEX_QUAL __device__\n'
                          '#define COMPLEX_NEED_FLOAT\n'
                          '#include "include/complex.h"')
    assert (tmp_path / "include" / complex_header.HEADER_NAME).is_file()
    assert "COMPLEX_NEED_DOUBLE" not in float_only


def test_opencl_gets_no_execution_space_qualifier(tmp_path):
    from pygpgpu.kernel import complex as complex_header

    line = complex_header.include_line(("COMPLEX_NEED_FLOAT",), "opencl", str(tmp_path))
    assert line.splitlines()[0] == "#define COMPLEX_QUAL "


def test_a_complex_value_is_recognized_by_its_dtype_alone():
    from pygpgpu.kernel.translator import DTYPE_MAP, infer_ctype, is_complex

    assert infer_ctype(np.complex64(1.0 + 2.0j)) == "c64"
    assert infer_ctype(np.complex128(1.0 + 2.0j)) == "c128"
    # A python complex has no width of its own, so it takes the one the python float beside
    # it takes: `float` by default, `double` under `@kernel(float=np.float64)`.
    assert infer_ctype(1.0 + 2.0j) == "c64"
    assert infer_ctype(1.0 + 2.0j, {"float": "double"}) == "c128"
    assert infer_ctype(np.zeros(2, dtype=np.complex64)) == ("*", "c64")
    assert infer_ctype(np.zeros(2, dtype=np.complex128)) == ("*", "c128")
    assert is_complex("c64") and is_complex("c128")
    assert not is_complex("float")
    assert DTYPE_MAP["complex64"] == "c64"
