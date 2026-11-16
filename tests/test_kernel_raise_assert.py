"""`raise` and `assert` must reach the caller as python's own exception.

A device launch reports one status word, so an abort from inside a work item used to arrive
as ``RuntimeError: CUDA_ERROR_LAUNCH_FAILED`` -- the type, the message and the line were all
gone, while the same source on the interpreter raised ``ValueError("must be positive")`` at
the right line.

The device cannot carry a python object, so the information travels as **data**: the raising
work item writes the line it raised from and a message id into a launch-scoped exception
slot, sets a "failed" flag, and every other work item stops at the next checkpoint. The host
then rebuilds the exception python would have raised, with its type, message and the caller's
traceback shape.

Only a **literal** message is supported -- an exception is logged from one place in the
source, and the text is what identifies that place. A message computed from a value cannot be
carried (there is no string in the device's type system), so it is refused at translation
time with a message that says so, rather than reaching the device and losing its text.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import get_global_id, kernel
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
# raise keeps python's type and message
# ---------------------------------------------------------------------------


@kernel(domain="out")
def raises_value_error(out: np.ndarray):
    i = get_global_id(0)
    if out[i] < 0.0:
        raise ValueError("the input must not be negative")
    out[i] = 1.0


@kernel(domain="out")
def raises_runtime_error(out: np.ndarray):
    i = get_global_id(0)
    if out[i] < 0.0:
        raise RuntimeError("bad state")
    out[i] = 1.0


@kernel(domain="out")
def raises_unconditionally(out: np.ndarray):
    raise ValueError("always fails")


def test_a_raised_exception_keeps_its_type_and_message(gpu_backend):
    """`ValueError("the input must not be negative")`, not a launch-failure RuntimeError."""
    out = np.array([-1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="the input must not be negative"):
        raises_value_error[gpu_backend][(1, 1)](out)


def test_a_different_exception_type_is_preserved(gpu_backend):
    """The class the body named is the class the caller catches."""
    out = np.array([-1.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match="bad state"):
        raises_runtime_error[gpu_backend][(1, 1)](out)


def test_a_raise_on_the_happy_path_does_not_fire(gpu_backend):
    """The control: a guard that does not trip leaves the launch untouched."""
    out = np.array([1.0], dtype=np.float32)
    raises_value_error[gpu_backend][(1, 1)](out)
    assert float(out[0]) == pytest.approx(1.0)


def test_an_unconditional_raise_always_reports(gpu_backend):
    out = np.array([0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="always fails"):
        raises_unconditionally[gpu_backend][(1, 1)](out)


# ---------------------------------------------------------------------------
# assert is the same mechanism
# ---------------------------------------------------------------------------


@kernel(domain="out")
def asserts_positive(out: np.ndarray):
    i = get_global_id(0)
    assert out[i] > 0.0, "must be positive"
    out[i] = 1.0


def test_an_assert_reaches_the_caller_as_assertion_error(gpu_backend):
    """`assert cond, "text"` raises `AssertionError` with that text, as python does."""
    out = np.array([-1.0], dtype=np.float32)
    with pytest.raises(AssertionError, match="must be positive"):
        asserts_positive[gpu_backend][(1, 1)](out)


def test_a_passing_assert_is_transparent(gpu_backend):
    out = np.array([2.0], dtype=np.float32)
    asserts_positive[gpu_backend][(1, 1)](out)
    assert float(out[0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# What is refused at translation time
# ---------------------------------------------------------------------------


def computed_message(out):
    for i in range(out.shape[0]):
        if out[i] < 0.0:
            raise ValueError("bad value: " + str(out[i]))
        out[i] = 1.0


def computed_fstring_message(out):
    for i in range(out.shape[0]):
        if out[i] < 0.0:
            raise ValueError(f"bad value {out[i]}")
        out[i] = 1.0


def no_message(out):
    """A bare `raise ValueError` carries no message: that is python-legal and supported."""
    for i in range(out.shape[0]):
        if out[i] < 0.0:
            raise ValueError
        out[i] = 1.0


@pytest.mark.parametrize("func", [computed_message, computed_fstring_message])
def test_a_message_that_is_not_a_literal_is_refused(func):
    """The message identifies the raise site, so it has to be fixed text."""
    with pytest.raises(SyntaxError, match="literal"):
        generate(func, np.zeros(1, np.float32))


def test_a_bare_exception_class_is_accepted():
    """`raise ValueError` with no message is ordinary python; the type is what matters."""
    source = generate(no_message, np.zeros(1, np.float32))
    assert "raise" in source.lower() or "exception" in source.lower()
