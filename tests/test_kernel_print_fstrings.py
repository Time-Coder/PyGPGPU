"""`print(f"...")`: an f-string is the print pieces python already has.

`print("slot", i, sep="")` and `print(f"slot {i}")` produce the same text, and the
translator already knew how to emit the first one: an f-string is a `JoinedStr` whose
literal runs are fixed text and whose `{expression}` fields are each one value, so the two
spellings expand to the same call sites. Only the joined form was refused, with
"unsupported expression: JoinedStr" -- which is the spelling a programmer reaches for
first when they want to label debug output.

Two things are deliberately still refused, because python's formatting machinery has no
device counterpart and silently ignoring them would print something else:

    f"{value!r}"     repr() is a python function
    f"{value:>4}"    a format spec is python's formatter

Both name the plain field to use instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.kernel.translator import FunctionTranslator
from pygpgpu.kernel.decorator import infer_ctype


def generate(func, *types, backend: str = "cuda") -> str:
    """The rendered device source for `func`, without compiling it.

    Accepts the decorated kernel or the plain function: the translator needs the source of
    the function itself.
    """
    raw = getattr(func, "raw_func", func)
    argument_types = [infer_ctype(t) for t in types]
    translator = FunctionTranslator(raw, argument_types, backend, lambda name: None, {},
                                    is_kernel=True)
    translator.translate()
    return "\n\n".join(source for _, source in translator.render_kernels("k_probe"))


# ---------------------------------------------------------------------------
# What the format string says
# ---------------------------------------------------------------------------


def labeled_field(out):
    """`f"slot {i}"`: literal text around one field."""
    for i in range(out.shape[0]):
        print(f"slot {i}")
        out[i] = i


def two_fields(out):
    """Two fields with text between them."""
    for i in range(out.shape[0]):
        print(f"[{i}] value = {i * 2}")
        out[i] = i


def text_only(out):
    """An f-string with no fields at all is just a string."""
    for i in range(out.shape[0]):
        print(f"plain text")  # noqa: F541 - the fieldless f-string is the case under test
        out[i] = i


def real_field(out):
    """A float field takes the real conversion, as a float argument does."""
    for i in range(out.shape[0]):
        print(f"x = {out[i] * 1.5}")
        out[i] = i


def fstring_beside_an_argument(out):
    """An f-string is one print argument; the separator still goes between arguments."""
    for i in range(out.shape[0]):
        print(f"slot {i}", i)
        out[i] = i


def conversation(out):
    """A conversion (`!r`) asks python's repr."""
    for i in range(out.shape[0]):
        print(f"slot {i!r}")
        out[i] = i


def format_spec(out):
    """A format spec asks python's formatter."""
    for i in range(out.shape[0]):
        print(f"slot {i:>4}")
        out[i] = i


def test_a_field_becomes_a_printf_conversion():
    """The text and the field land in one format string, in order."""
    source = generate(labeled_field, np.zeros(2, np.float32))
    assert 'printf("slot %d' in source


def test_text_and_fields_keep_their_order():
    """Two fields with the literal text between them."""
    source = generate(two_fields, np.zeros(2, np.float32))
    assert "[%d] value = %d" in source


def test_a_fieldless_fstring_is_plain_text():
    """No fields means no conversion, exactly like a plain string argument."""
    source = generate(text_only, np.zeros(2, np.float32))
    assert '"plain text' in source
    assert "%d" not in source.split('"plain text')[1].split('"')[0]


def test_a_real_field_takes_the_real_conversion():
    """A float value is printed the way a float argument is, not as an integer."""
    source = generate(real_field, np.zeros(2, np.float32))
    assert "x = " in source
    assert "%d" not in source.split("x = ")[1].split('"')[0]


def test_an_fstring_beside_another_argument_keeps_the_separator():
    """`print(f"slot {i}", i)` separates the two arguments with a space."""
    source = generate(fstring_beside_an_argument, np.zeros(2, np.float32))
    assert "slot %d " in source


def test_a_conversion_is_refused_with_the_reason():
    """`!r` is python's repr, which the device cannot call."""
    with pytest.raises(SyntaxError) as refused:
        generate(conversation, np.zeros(2, np.float32))
    message = str(refused.value)
    assert "conversion" in message
    assert "expression" in message


def test_a_format_spec_is_refused_with_the_reason():
    """`:>4` is python's formatter, which the device does not have."""
    with pytest.raises(SyntaxError) as refused:
        generate(format_spec, np.zeros(2, np.float32))
    message = str(refused.value)
    assert "format spec" in message


# ---------------------------------------------------------------------------
# The backend that actually drains the buffer
# ---------------------------------------------------------------------------


def test_the_opencl_source_carries_the_same_format_and_the_launch_succeeds():
    """OpenCL is the backend whose runtime drains these call sites.

    The drained text is the runtime's own output buffer, which no host-side accessor here
    reads, so the format string in the generated source is what can be checked -- and it is
    also exactly what the device prints: the device has no other way to render the line.
    The launch is run as well, so an emission that compiled but was malformed would show up.
    """
    from pygpgpu.kernel.decorator import _probe_backend

    if not _probe_backend("opencl"):
        pytest.skip("the opencl runtime is what drains the printf call sites")

    source = generate(labeled_field, np.zeros(2, np.float32), backend="opencl")
    assert 'printf("slot %d' in source

    @kernel
    def printer(out: np.ndarray):
        for i in range(out.shape[0]):
            print(f"item {i}")
            out[i] = i

    out = np.zeros(4, np.float32)
    printer["opencl"](out)
    np.testing.assert_allclose(out, np.arange(4, dtype=np.float32))
