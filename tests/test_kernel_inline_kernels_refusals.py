"""What `with inline, kernel(...):` cannot mean, refused before anything is translated.

Every refusal here is a `SyntaxError` raised at capture time -- while the `with` head is being
entered, before the body has been read by the translator -- so the message can name the fix
rather than report a syntax error the user never believed they wrote. Error *positions* and
block *placement* live in `test_kernel_inline_kernels_placement.py`; the launch side of the
feature (data capture, work size, caching) is pinned in `test_kernel_inline_kernels.py`.
"""
from __future__ import annotations

import ast

import numpy as np
import pytest

from pygpgpu import inline, kernel
from pygpgpu.kernel.inline_kernels import site


def test_a_block_inside_a_kernel_body_is_refused()->None:
    """A `with inline` cannot live inside a `@kernel` body: that body IS device code.

    Left to the translator, the two backends disagreed -- the interpreter ran the `with` as
    plain python and produced a different answer, while the device backends refused it with
    `unsupported statement: With`. One spelling must mean one thing on every backend, so the
    capture refuses it by name.

    The refusal happens when the *outer* kernel runs, because that is when the body's `with`
    is reached; defining the kernel is not enough.
    """
    from pygpgpu import kernel as decorated_kernel

    out:np.ndarray = np.zeros(4, dtype=np.float32)

    def make() -> object:
        @decorated_kernel(default_device="python")
        def outer(target):
            for i in range(target.shape[0]):
                with inline, kernel(default_device="python"):
                    target[i] = 1.0
        return outer

    outer = make()
    with pytest.raises(SyntaxError, match="inside a @kernel body"):
        outer(out)


def test_a_block_inside_a_kernel_body_is_refused_on_every_backend(backend)->None:
    """The refusal happens at capture, so no backend can reach a different answer first.

    The interpreter reaches the `with` while running the body, so the capture refuses it
    there; a device backend never runs the body, so its translator refuses the same spelling
    in its own words. Both refuse, and neither computes a different number.
    """
    source:str = (
        "import numpy as np\n"
        "from pygpgpu import inline, kernel\n"
        "\n"
        "@kernel\n"
        "def outer(out):\n"
        "    for i in range(out.shape[0]):\n"
        "        with inline, kernel():\n"
        "            out[i] = 1.0\n")
    module_path = _write_module(source)
    try:
        module = _import_module(module_path)
        out:np.ndarray = np.zeros(4, dtype=np.float32)
        with pytest.raises(SyntaxError, match="launch site|inside a @kernel body"):
            module.outer[backend](out)
    finally:
        module_path.unlink(missing_ok=True)


def _write_module(source:str):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp(prefix="pygpgpu_inline_refusal_")) / "block_in_kernel.py"
    path.write_text(source, encoding="utf-8")
    return path


def _import_module(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inline_alone_is_refused()->None:
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with pytest.raises(SyntaxError, match="kernel"):
        with inline:
            out[0] = 1.0


def test_a_head_without_inline_is_refused()->None:
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with pytest.raises(Exception, match="inline"):
        with kernel(default_device="python"):
            out[0] = 1.0


def test_inline_must_come_first()->None:
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with pytest.raises(Exception, match="inline"):
        with kernel(default_device="python"), inline:
            out[0] = 1.0


def test_a_nonlocal_in_a_block_is_refused()->None:
    holder:int = 0

    def run()->None:
        nonlocal holder
        with inline, kernel(default_device="python"):
            nonlocal holder
            holder = 1

    with pytest.raises(SyntaxError, match="nonlocal"):
        run()


def test_a_one_line_block_is_refused()->None:
    """The block has to be on its own lines: a one-line body has no text to rebuild from.

    The one-line spelling cannot itself be written inside this test -- python parses it into a
    real one-line `with` body -- so the refusal is pinned against the same predicate the
    capture uses, over a parsed one-line statement.
    """
    source:str = (
        "def f(out):\n"
        "    with inline, kernel():\n"
        "        pass\n"
        "    with inline, kernel(): out[0] = 1.0\n")
    node:ast.With = [n for n in ast.walk(ast.parse(source))
                     if isinstance(n, ast.With)][-1]
    assert node.body[0].lineno == node.lineno
    with pytest.raises(SyntaxError, match="own lines"):
        site._refuse_a_one_line_body(node)  # type: ignore[attr-defined]


def test_a_shape_domain_on_the_decorator_is_refused()->None:
    """A decorator is written once and called many times: only a parameter name survives."""
    with pytest.raises(SyntaxError, match="string"):
        @kernel(domain=(4, 256))
        def add(a, out):
            for i in range(out.shape[0]):
                out[i] = a[i]


def test_domain_and_the_work_size_pair_are_refused_together()->None:
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with pytest.raises(SyntaxError, match="domain"):
        with inline, kernel(default_device="python", domain="out",
                            threads_per_block=4):
            for i in range(out.shape[0]):
                out[i] = 1.0
