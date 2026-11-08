"""The printing tests, split out of `test_metal_codegen.py`.

The subject is printing.
"""
from __future__ import annotations

from _helpers.test_metal_codegen_helpers import (
    generate,
    printer,
)

import pytest


def test_print_is_refused_on_metal_and_still_works_on_opencl() -> None:
    """Metal's printf goes to a log no host code reads, so the call site is refused.

    The generated call sites write into a per-call-site output buffer that the
    OpenCL runtime drains. Metal has no such buffer, so emitting them would compile
    and silently lose the output -- the same reason HIP refuses its own.
    """
    with pytest.raises(SyntaxError) as refused:
        generate(printer, 2)
    message = str(refused.value)
    assert "metal" in message
    assert "print()" in message

    # The backends that do drain the buffer keep generating it.
    for backend in ("opencl", "cuda"):
        source = generate(printer, 2, backend)
        assert "printf" in source
