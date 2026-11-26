"""Where a block may be written, and where a translation error points.

A block is a statement of the caller's file, so it has to work at module level, inside a
`try`/`finally`, and inside a generator -- the places a `with` statement is ordinary. And
because a failing body is reported as a translation error rather than a traceback, the
position it carries has to be the block's own file and line, with the real source text under
the caret. That is what this file pins.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pygpgpu import inline, kernel


def _close(actual:object, expected:object)->None:
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def _import_file(name:str, path:Path)->object:
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ERROR_POSITION_SOURCE:str = '''
"""A block reading a name the module does not define: the translation must blame line 11."""
import numpy as np
from pygpgpu import inline, kernel

out = np.zeros(4, dtype=np.float32)

try:
    with inline, kernel(default_device="{backend}"):
        for i in range(out.shape[0]):
            out[i] = not_a_kernel_name
except SyntaxError as error:
    position = (error.filename, error.lineno, error.text, str(error))
'''


def test_a_translation_error_points_at_the_block_line(gpu_backend, tmp_path:Path)->None:
    """The reported line is the block's own line in the user's file, with its real text."""
    path:Path = tmp_path / "inline_error_position.py"
    path.write_text(_ERROR_POSITION_SOURCE.format(backend=gpu_backend), encoding="utf-8")
    module = _import_file("inline_error_position", path)
    filename, lineno, text, message = module.position
    assert Path(filename).resolve() == path.resolve()
    assert lineno == 11
    assert text is not None and "not_a_kernel_name" in text
    assert "not_a_kernel_name" in message


_MODULE_BLOCK_SOURCE:str = '''
"""A block written at module level: the module namespace is the whole scope."""
import numpy as np
from pygpgpu import inline, kernel

data = np.arange(4, dtype=np.float32)
result = np.zeros(4, dtype=np.float32)

with inline, kernel(default_device="python"):
    for i in range(result.shape[0]):
        result[i] = data[i] * 2.0
'''


def test_a_module_level_block_launches_at_import(tmp_path:Path)->None:
    path:Path = tmp_path / "inline_module_block.py"
    path.write_text(_MODULE_BLOCK_SOURCE, encoding="utf-8")
    module = _import_file("inline_module_block", path)
    _close(module.result, module.data * 2.0)


def test_a_block_in_a_try_finally_runs_and_the_finally_sees_the_result(backend)->None:
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    seen:np.ndarray = np.zeros(1, dtype=np.float32)
    try:
        with inline, kernel(default_device=backend):
            for i in range(out.shape[0]):
                out[i] = a[i] + 1.0
    finally:
        seen[0] = float(out[0])
    _close(out, a + 1.0)
    assert seen[0] == pytest.approx(float(a[0]) + 1.0)


def test_a_block_inside_a_generator_launches_at_first_next(backend)->None:
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    a:np.ndarray = np.arange(4, dtype=np.float32)

    def generate():
        with inline, kernel(default_device=backend):
            for i in range(out.shape[0]):
                out[i] = a[i] + 1.0
        yield 1

    assert list(generate()) == [1]
    _close(out, a + 1.0)
