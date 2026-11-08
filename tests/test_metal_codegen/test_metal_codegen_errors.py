"""The errors tests, split out of `test_metal_codegen.py`.

The subject is errors.
"""
from __future__ import annotations

from _helpers.test_metal_codegen_helpers import (
    _compile_with_stub,
    body_with_unknown_call,
)

import numpy as np
import pytest
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.translator import FunctionTranslator


def test_a_translation_error_names_the_file_and_line() -> None:
    """A refused body reports where the offending name is, not a generated excerpt.

    The translator parses a copy of the function's source, so a position it reports
    has to be mapped back to the file the user wrote. The check below triggers the
    commonest refusal -- an unknown function in the body -- and asserts the message
    carries both the name and the real line of the file.
    """
    import inspect

    line_of_call = None
    source_lines, first_line = inspect.getsourcelines(body_with_unknown_call)
    for offset, text in enumerate(source_lines):
        if "unknown_device_function" in text:
            line_of_call = first_line + offset
            break
    assert line_of_call is not None

    with pytest.raises(SyntaxError) as refused:
        _compile_with_stub(body_with_unknown_call)
    message = str(refused.value)

    assert "unknown_device_function" in message
    # The reported position is the caller's file and the line the name sits on.
    assert f"line {line_of_call}" in message, message
    assert "test_metal_codegen" in message


@pytest.mark.parametrize("case,expectation", [
    ("print", ("opencl or cuda", "print()")),
    ("image_read", ("cuda, opencl or python", "images (read)")),
    ("image_write", ("cuda, opencl or python", "images (write)")),
], ids=["print", "image read", "image write"])
def test_a_refusal_names_a_usable_alternative(case, expectation) -> None:
    """Every refusal says why, and which backends do work.

    A message that only said "not supported" would leave a caller with nowhere to go.
    Each of these carries both halves: the reason the generated code would be wrong,
    and the backends that implement it. The pair is checked together because either
    half alone makes the message unhelpful.
    """
    def print_body(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i]
            print(x[i])

    def image_body(img, out):
        for i in range(out.shape[0]):
            out[i] = img[i, 0]

    def image_write_body(img, out):
        for i in range(out.shape[0]):
            img[i, 0] = out[i]

    array = np.zeros(8, np.float32)
    translator = FunctionTranslator(
        {"print": print_body, "image_read": image_body,
         "image_write": image_write_body}[case],
        [("image", 4, "float"), infer_ctype(array)] if case != "print"
        else [infer_ctype(array)] * 2,
        "metal", lambda name: None, {}, is_kernel=True)
    with pytest.raises(SyntaxError) as refused:
        translator.translate()
    message = str(refused.value)
    reason, backup = expectation
    assert reason in message, message[:120]
    assert backup in message, message[:120]
