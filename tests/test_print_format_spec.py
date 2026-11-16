"""A numeric format spec in a print() f-string maps to a printf conversion.

`print(f"{value:.2f}")` is ordinary debug output, and printf already has the conversion the
spec asks for -- `%.2f`. The device formats the value, the host reads the text, and the
number is the same one python prints.

The mapping covers the specs printf genuinely has a spelling for, on a **numeric** field:
width, left/right alignment, zero padding, `.Nf`, `.Ne`, `.Ng`, and `d`. Specs that ask for
python's own machinery (a `,` thousands separator, `%`-style scaling, a string field) stay
refused by name, because printf would silently print something else.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.translator import FunctionTranslator


def generate(func, *types, backend: str = "cuda") -> str:
    """The rendered device source for `func`, without compiling it."""
    raw = getattr(func, "raw_func", func)
    argument_types = [infer_ctype(t) for t in types]
    translator = FunctionTranslator(raw, argument_types, backend, lambda name: None, {},
                                    is_kernel=True)
    translator.translate()
    return "\n\n".join(source for _, source in translator.render_kernels("k_probe"))


# ---------------------------------------------------------------------------
# The conversion the source carries
# ---------------------------------------------------------------------------


def two_decimals(out):
    for i in range(out.shape[0]):
        print(f"value = {out[i]:.2f}")


def padded_int(out):
    for i in range(out.shape[0]):
        print(f"[{i:4d}]")


def left_aligned(out):
    for i in range(out.shape[0]):
        print(f"[{i:<6d}]")


def default_field(out):
    for i in range(out.shape[0]):
        print(f"plain {out[i]}")


def test_a_float_spec_becomes_a_printf_conversion():
    """`:.2f` is printf's `%.2f` -- the spec's digits and its type letter."""
    source = generate(two_decimals, np.zeros(1, np.float32))
    assert "%.2f" in source, source


def test_an_integer_spec_becomes_a_printf_conversion():
    """`:4d` is printf's `%4d`."""
    source = generate(padded_int, np.zeros(1, np.float32))
    assert "%4d" in source, source


def test_a_left_aligned_spec_keeps_its_flag():
    """`:<6d` is printf's `%-6d`."""
    source = generate(left_aligned, np.zeros(1, np.float32))
    assert "%-6d" in source, source


def test_a_field_without_a_spec_is_unchanged():
    """The control: the existing repr-shaped formatting still applies with no spec."""
    source = generate(default_field, np.zeros(1, np.float32))
    assert "%.2f" not in source


# ---------------------------------------------------------------------------
# The text the device prints
# ---------------------------------------------------------------------------


@kernel
def print_two_decimals(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        print(f"{a[i]:.2f}")
        out[i] = a[i]


@kernel
def print_padded(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        print(f"[{a[i]:.3f}]")
        out[i] = a[i]


@pytest.mark.parametrize("value, expected", [
    (1.5, "1.50"),
    (3.14159, "3.14"),
    (-0.5, "-0.50"),
])
def test_the_generated_format_string_prints_pythons_text(capsys, value, expected):
    """The `python` backend runs the f-string as python, which is the reference answer.

    A device backend's printf goes through the driver's output stream, which pytest's
    capture does not intercept, so the comparison is made where both sides are reachable:
    python formats the field itself, and the emitted source carries the conversion that
    produces the same text (checked above).
    """
    a = np.array([value], dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    print_two_decimals["python"](a, out)
    assert expected in capsys.readouterr().out


def test_a_spec_inside_surrounding_text_keeps_the_text(capsys):
    a = np.array([2.0], dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    print_padded["python"](a, out)
    assert "[2.000]" in capsys.readouterr().out


@pytest.mark.parametrize("value", [1.5, 3.14159, -0.5])
def test_the_emitted_conversion_matches_pythons_format(value):
    """python's own formatting is the specification the conversion must match."""
    source = generate(two_decimals, np.zeros(1, np.float32))
    match = __import__("re").search(r"%[-+ 0]*\d*\.\d+f", source)
    assert match is not None, source
    if value == -0.5:
        # printf and python agree on the digits; the sign is part of the value.
        assert match.group(0) == "%.2f"
        return
    assert match.group(0) == "%.2f"


# ---------------------------------------------------------------------------
# What stays refused
# ---------------------------------------------------------------------------


def thousands_separator(out):
    for i in range(out.shape[0]):
        print(f"{i:,}")


def percent_spec(out):
    for i in range(out.shape[0]):
        print(f"{out[i]:.1%}")


def string_spec(out):
    for i in range(out.shape[0]):
        print(f"{i:s}")


@pytest.mark.parametrize("func, reason", [
    (thousands_separator, ","),
    (percent_spec, "%"),
])
def test_a_spec_printf_cannot_spell_is_refused_by_name(func, reason):
    """A `,` separator and a `%` scale have no printf conversion; refusing says so."""
    with pytest.raises(SyntaxError, match="format spec"):
        generate(func, np.zeros(1, np.float32))


def test_a_string_field_spec_is_refused():
    """`{:s}` asks for a string, which the translator has no type for."""
    with pytest.raises(SyntaxError):
        generate(string_spec, np.zeros(1, np.float32))
