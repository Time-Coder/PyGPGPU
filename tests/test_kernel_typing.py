"""A `@kernel` keeps the decorated function's parameter list, for an editor and a checker.

`@kernel` returns a callable object, so what an editor shows on a call comes from the
*type* of the object rather than from the function that was decorated. Publishing plain
``PyKernel`` as the decorator's result made a checker read the decorated name as the
decorator itself (``(f) -> PyKernel``), so every call was checked against
``__call__(*args, **kwargs)``: no parameter names in a hint, and no error for a call that
passes the wrong number of arguments.

What fixes it is PEP 612: ``PyKernel`` is generic over the decorated function's own
parameters, and both decorator spellings are declared as overloads so the function's
signature lands on the decorated name. A user-defined generic over a ParamSpec needs
Python 3.10+, and this package supports 3.7+, so the generic base exists only for the
type checkers while the runtime keeps a plain class that answers ``PyKernel[_P]`` with
itself -- subscripting the class at import must never fail.

These tests pin the runtime half: the annotations are in place, the base is the one the
checkers see as generic, subscripting works on every interpreter, and decoration still
behaves exactly as before. The editor-visible half is what the annotations produce, and
`python check.py` (which runs `ty`) is what keeps it honest.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable

import numpy as np
import pytest

from pygpgpu import kernel


def _decorator_module()->Any:
    """The module the decorator lives in, by path rather than by attribute access.

    ``pygpgpu.kernel`` is the decorator at the top level and the subpackage below it, so
    ``import pygpgpu.kernel.decorator`` binds the wrong name in some spellings.
    """
    return importlib.import_module("pygpgpu.kernel.decorator")


# ---------------------------------------------------------------------------
# The typing hooks the checkers read
# ---------------------------------------------------------------------------


def test_paramspec_is_declared_for_both_roles()->None:
    """The parameter list and the result type are separate type variables."""
    module = _decorator_module()
    assert module._P.__class__.__name__ == "ParamSpec"
    assert module._R.__class__.__name__ == "TypeVar"


def test_kernel_base_is_generic_where_a_checker_looks()->None:
    """The runtime base is not ``Generic``; the annotated one is.

    A user-defined generic over a ParamSpec is not available on 3.7, so the package
    declares it inside ``if TYPE_CHECKING`` and keeps a plain class at runtime. The
    annotated branch still names ``Generic``, or a checker would see an ordinary class
    and lose the parameters again.
    """
    module = _decorator_module()
    source:str = inspect.getsource(module)
    marker:str = "if TYPE_CHECKING:\n    class _KernelBase(Generic[_P]):"
    assert marker in source, "the TYPE_CHECKING branch must declare the generic base as a class"


def test_the_runtime_base_answers_a_subscript()->None:
    """``PyKernel[_P]`` is spelled in annotations, so it has to evaluate at runtime.

    ``from __future__ import annotations`` means the annotations are not evaluated here,
    but a user's own ``def f() -> PyKernel[...]`` or a ``cast`` may be, and the class is
    subscripted the moment somebody writes ``PyKernel[int]``.
    """
    from pygpgpu.kernel.decorator import PyKernel

    assert PyKernel[object] is PyKernel
    assert PyKernel[Any] is PyKernel


@pytest.mark.parametrize("cls_name", ["PyKernel", "_KernelCall", "_BoundKernelMethod"])
def test_every_call_surface_is_generic(cls_name:str)->None:
    """Each form a caller reaches (plain, bracketed, bound) carries the parameters.

    A bracket or a receiver selects a backend or drops `self`; neither changes which
    arguments the launch takes, so each wrapper is generic over the same ``_P``.
    """
    module = _decorator_module()
    cls:type = getattr(module, cls_name)
    base:type = module._KernelBase
    assert base in cls.__mro__, f"{cls_name} must inherit the parameterized base"


# ---------------------------------------------------------------------------
# Decoration itself is unchanged
# ---------------------------------------------------------------------------


@kernel
def _plain(a: np.ndarray, b: np.ndarray, out: np.ndarray, alpha: float = 2.0)->None:
    for i in range(out.shape[0]):
        out[i] = a[i] * alpha + b[i]


def test_decoration_still_publishes_a_kernel()->None:
    from pygpgpu.kernel.decorator import PyKernel

    assert isinstance(_plain, PyKernel)
    assert _plain.__name__ == "_plain"
    assert _plain.is_pykernel is True


def test_the_runtime_signature_is_the_authors_own()->None:
    """The editor reads the annotations; ``inspect`` reads the real function.

    Both have to keep agreeing after the typing change, because the argument check at the
    call boundary binds against this signature.
    """
    signature:inspect.Signature = inspect.signature(_plain.raw_func)
    assert list(signature.parameters) == ["a", "b", "out", "alpha"]
    assert signature.parameters["alpha"].default == 2.0


def test_a_wrong_call_still_raises_pythons_own_error()->None:
    """The typing is for the editor; the boundary still words the error itself."""
    a = np.zeros(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    misuse:Callable[..., Any] = _plain
    # CPython's own wording, naming the argument the caller got wrong -- not a message
    # this module writes. The shape of the call is checked before any argument is
    # translated, which is what makes a wrong call cost nothing.
    with pytest.raises(TypeError, match="unexpected keyword argument 'extra'"):
        misuse(a, b, extra=1)


def test_the_bracket_form_still_runs()->None:
    """``k[selector]`` is untouched: it returns a wrapper that launches as before."""
    a = np.arange(4, dtype=np.float32)
    b = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    _plain["python"](a, b, out, 2.0)
    assert np.allclose(out, a * 2.0 + b)
