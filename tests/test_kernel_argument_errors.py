"""Kernel argument-binding errors must read exactly like the undecorated function's.

A `@kernel` call is checked against the author's own signature, so a wrong call has to
fail the way the same call to the plain function fails: same wording, same plural forms,
same "Did you mean ...?" suggestion, and the kernel's own qualified name in front. These
tests compare the two messages directly rather than asserting a fixed string, so they keep
following the running interpreter instead of pinning one version's phrasing.

Every call here is deliberately wrong -- that is the subject of the test -- so the plain
functions are reached through :func:`_misuse`, which tells a type checker that the wrong
call is intentional instead of flagging each line.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pytest

from pygpgpu import kernel


def _misuse(func: Callable[..., Any]) -> Callable[..., Any]:
    """The function as an untyped callable: these tests call it wrongly on purpose."""
    return func


# ---------------------------------------------------------------------------
# Plain functions and their kernel twins: the signatures must match exactly,
# because CPython counts the positional arguments it was given.
# ---------------------------------------------------------------------------


def plain_three(a, b, c):
    return None


@kernel
def kernel_three(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    for i in range(b.shape[0]):
        b[i] = a[i]


def plain_default(a, b, c=3.0):
    return None


@kernel
def kernel_default(a: np.ndarray, b: np.ndarray, c: float = 3.0):
    for i in range(b.shape[0]):
        b[i] = a[i] * c


def plain_keyword_only(a, b, *, k):
    return None


@kernel
def kernel_keyword_only(a: np.ndarray, b: np.ndarray, *, k: int):
    for i in range(b.shape[0]):
        b[i] = a[i] * k


class PlainModel:
    def run(self, out, x):
        return None


class KernelModel:
    @kernel
    def run(self, out: np.ndarray, x: int):
        for i in range(out.shape[0]):
            out[i] = x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _message(thunk: Callable[[], Any], name: str) -> str:
    """The TypeError text of a bad call, with the function's own name normalised away."""
    with pytest.raises(TypeError) as caught:
        thunk()
    return str(caught.value).replace(name + "()", "<kernel>()")


def _compare(plain_thunk: Callable[[], Any], plain_name: str,
             kernel_thunk: Callable[[], Any], kernel_name: str) -> None:
    assert _message(kernel_thunk, kernel_name) == _message(plain_thunk, plain_name)


def _target(kernel_object: Any, backend: str) -> Any:
    """The kernel bound to ``backend`` (the python backend has no device to select)."""
    return kernel_object["python"] if backend == "python" else kernel_object[backend]


# ---------------------------------------------------------------------------
# Positional arity
# ---------------------------------------------------------------------------


def test_too_many_positional_arguments(backend):
    x = np.zeros(4, dtype=np.float32)
    plain, kern = _misuse(plain_three), _target(kernel_three, backend)
    _compare(lambda: plain(1, 2, 3, 4), "plain_three",
             lambda: kern(x, x, x, 4), "kernel_three")


def test_too_few_positional_arguments(backend):
    x = np.zeros(4, dtype=np.float32)
    plain, kern = _misuse(plain_three), _target(kernel_three, backend)
    _compare(lambda: plain(1), "plain_three",
             lambda: kern(x), "kernel_three")


def test_too_many_with_a_defaulted_parameter(backend):
    x = np.zeros(4, dtype=np.float32)
    plain, kern = _misuse(plain_default), _target(kernel_default, backend)
    _compare(lambda: plain(1, 2, 3.0, 4.0), "plain_default",
             lambda: kern(x, x, 3.0, 4.0), "kernel_default")


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------


def test_unexpected_keyword_argument(backend):
    x = np.zeros(4, dtype=np.float32)
    plain, kern = _misuse(plain_three), _target(kernel_three, backend)
    _compare(lambda: plain(1, 2, c=3, zz=1), "plain_three",
             lambda: kern(x, x, c=x, zz=1), "kernel_three")


def test_unexpected_keyword_suggestion_is_kept(backend):
    """A near-miss keyword must still get the interpreter's own suggestion."""
    x = np.zeros(4, dtype=np.float32)
    plain, kern = _misuse(plain_three), _target(kernel_three, backend)
    plain_message: str = _message(lambda: plain(1, 2, cc=3), "plain_three")
    assert _message(lambda: kern(x, x, cc=x), "kernel_three") == plain_message
    assert "unexpected keyword argument 'cc'" in plain_message


def test_multiple_values_for_an_argument(backend):
    x = np.zeros(4, dtype=np.float32)
    plain, kern = _misuse(plain_three), _target(kernel_three, backend)
    _compare(lambda: plain(1, a=2, b=3, c=4), "plain_three",
             lambda: kern(x, a=x, b=x, c=x), "kernel_three")


def test_missing_keyword_only_argument(backend):
    x = np.zeros(4, dtype=np.float32)
    plain, kern = _misuse(plain_keyword_only), _target(kernel_keyword_only, backend)
    _compare(lambda: plain(1, 2), "plain_keyword_only",
             lambda: kern(x, x), "kernel_keyword_only")


# ---------------------------------------------------------------------------
# Methods: the name is the class-qualified one, and the receiver is counted like
# python counts it
# ---------------------------------------------------------------------------


def test_method_too_many_arguments(backend):
    out = np.zeros(4, dtype=np.float32)
    plain = _misuse(PlainModel().run)
    kern = _target(KernelModel().run, backend)
    _compare(lambda: plain(1, 2, 3), "PlainModel.run",
             lambda: kern(out, 2, 3), "KernelModel.run")


def test_method_too_few_arguments(backend):
    plain = _misuse(PlainModel().run)
    kern = _target(KernelModel().run, backend)
    _compare(lambda: plain(), "PlainModel.run",
             lambda: kern(), "KernelModel.run")


# ---------------------------------------------------------------------------
# The message must name the kernel, never a generic placeholder
# ---------------------------------------------------------------------------


def test_the_error_names_the_kernel(backend):
    x = np.zeros(4, dtype=np.float32)
    kern = _target(kernel_three, backend)
    with pytest.raises(TypeError) as caught:
        kern(x, x, x, 4)
    message: str = str(caught.value)
    assert message.startswith("kernel_three()")
    assert "positional arguments" in message


# ---------------------------------------------------------------------------
# An annotation is never a constraint on the argument
# ---------------------------------------------------------------------------
#
# The device type of an argument is decided by the *value* at call time (see
# `infer_ctype`), so a parameter's annotation is inert: it is neither read to choose a
# width nor checked against the argument. This used to be enforced for the two plain
# spellings (`np.ndarray` / `int`), and the check made a value's acceptance depend on
# which words the author happened to write: `k([1, 2, 3], out)` worked for an
# unannotated parameter, was refused for an `np.ndarray` one, and was accepted again
# under the `NDArray[np.float32]` spelling every example uses. A wrong value now fails
# where its type is actually used -- the launch path or the kernel body -- which is one
# rule for one value rather than three.


def test_a_sequence_reaches_an_ndarray_annotated_parameter(gpu_backend):
    """A list is an array argument whether or not the parameter says `np.ndarray`.

    The conversion of a numeric sequence happens in the launch path, from the value; the
    annotation used to reject the same list before it got there. (The interpreter backend
    is excluded for an unrelated reason: it runs the body unchanged, so a list that the
    body indexes with `.shape` fails there whatever the annotation says.)
    """
    x = np.zeros(4, dtype=np.float32)
    kern = _target(kernel_three, gpu_backend)
    assert kern(x, [1.0, 2.0, 3.0, 4.0], x) is None


def test_a_scalar_where_an_array_is_annotated_fails_at_the_launch(gpu_backend):
    """The failure names the parameter where its type is actually used -- the kernel body.

    This is the cost of ignoring annotations: a scalar passed for an array parameter is
    caught by the translator ("'b' is not an array parameter of this kernel") rather than
    by a call-site check that read the annotation.
    """
    x = np.zeros(4, dtype=np.float32)
    kern = _target(kernel_three, gpu_backend)
    with pytest.raises(SyntaxError) as caught:
        kern(x, 5, x)
    assert "not an array parameter" in str(caught.value)


def test_submit_and_async_call_use_the_same_binding(backend):
    """The non-blocking entry points bind arguments the same way, so they fail alike."""
    x = np.zeros(4, dtype=np.float32)
    kern = _target(kernel_three, backend)
    expected: str = _message(lambda: _misuse(plain_three)(1, 2, 3, 4), "plain_three")
    with pytest.raises(TypeError) as submit_error:
        kern.submit(x, x, x, 4)
    with pytest.raises(TypeError) as async_error:
        kern.async_call(x, x, x, 4)
    assert str(submit_error.value).replace("kernel_three()", "<kernel>()") == expected
    assert str(async_error.value).replace("kernel_three()", "<kernel>()") == expected


# ---------------------------------------------------------------------------
# The traceback must show the caller, not this library
# ---------------------------------------------------------------------------


def _library_frames(error: BaseException) -> "list[str]":
    """The frames of the pygpgpu package in the error's traceback (never this test file)."""
    import pygpgpu

    package_root = os.path.dirname(os.path.abspath(pygpgpu.__file__))
    frames: "list[str]" = []
    cursor: Any = error.__traceback__
    while cursor is not None:
        filename: str = cursor.tb_frame.f_code.co_filename
        if os.path.abspath(filename).startswith(package_root + os.sep):
            frames.append(f"{os.path.basename(filename)}:{cursor.tb_lineno}")
        cursor = cursor.tb_next
    return frames


def _raise_misuse(thunk) -> BaseException:
    """The error a wrong call raises, caught so its traceback can be inspected."""
    try:
        thunk()
    except BaseException as error:  # noqa: BLE001 - the traceback is the subject here
        return error
    raise AssertionError("the call was expected to fail")


def test_a_bad_call_has_no_library_frames(backend):
    """Native python blames the call site only; a decorated call must read the same."""
    x = np.zeros(4, dtype=np.float32)
    kern = _target(kernel_three, backend)
    error = _raise_misuse(lambda: kern(x, x, x, 4))
    assert isinstance(error, TypeError)
    assert _library_frames(error) == []


def test_a_bad_bracketed_call_has_no_library_frames(backend):
    """`k["cuda"](...)` goes through the bracket wrapper and must be trimmed too."""
    x = np.zeros(4, dtype=np.float32)
    kern = _target(kernel_three, backend)
    error = _raise_misuse(lambda: kern(x, x, x, 4))
    assert _library_frames(error) == []


def test_a_bad_method_call_has_no_library_frames(backend):
    out = np.zeros(4, dtype=np.float32)
    kern = _target(KernelModel().run, backend)
    error = _raise_misuse(lambda: kern(out, 2, 3))
    assert isinstance(error, TypeError)
    assert _library_frames(error) == []


def test_the_plain_function_and_the_kernel_have_the_same_frame_count(backend):
    """The same mistake must produce the same traceback shape either way."""
    x = np.zeros(4, dtype=np.float32)
    plain, kern = _misuse(plain_three), _target(kernel_three, backend)

    def frames(error: BaseException) -> int:
        count: int = 0
        cursor: Any = error.__traceback__
        while cursor is not None:
            count += 1
            cursor = cursor.tb_next
        return count

    plain_error = _raise_misuse(lambda: plain(1, 2, 3, 4))
    kernel_error = _raise_misuse(lambda: kern(x, x, x, 4))
    assert frames(kernel_error) == frames(plain_error)
