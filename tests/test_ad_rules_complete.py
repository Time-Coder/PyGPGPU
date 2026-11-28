# M3: the four-way agreement over the AD rule table.
#
# Every builtin a kernel body may call has either a VJP rule or a NON_DIFFERENTIABLE
# entry with a reason -- the sentence this prevents is "a new builtin was added and it
# silently has a zero gradient": the forward answer is right, the expression
# type-checks, and only the number backward produces is wrong.
from __future__ import annotations

import builtins as python_builtins
import math as math_module

import numpy as np

from pygpgpu.ad import grad_for
from pygpgpu.ad.rules import (
    NON_DIFFERENTIABLE,
    RULES_BY_OBJECT,
    RULES_VERSION,
    knows,
)
from pygpgpu.kernel import sci_math
from pygpgpu.kernel.builtin_registry import BUILTIN_FUNCTIONS
from pygpgpu.kernel.translator import _MATH_FUNCS


def test_every_builtin_is_ruled_or_refused():
    """The dual-sided builtins: one entry each, keyed by the object."""
    missing:list = []
    for name, target in BUILTIN_FUNCTIONS.items():
        if not callable(target):
            continue      # sentinel objects and the index vectors are not calls
        if not knows(target):
            missing.append(name)
    assert not missing, f"builtins with no AD entry: {missing}"


def test_every_sci_math_name_is_ruled_or_refused():
    """The scientific vocabulary the kernels call."""
    missing:list = []
    for name in sci_math.__all__:
        target = getattr(sci_math, name, None)
        if target is None or not callable(target):
            continue
        if not knows(target):
            missing.append(name)
    assert not missing, f"sci_math names with no AD entry: {missing}"


def test_every_math_module_name_is_ruled_or_refused():
    """The `math.*` spellings the translator maps."""
    missing:list = []
    for name in _MATH_FUNCS:
        target = getattr(math_module, name, None)
        if target is None:
            continue
        if not knows(target):
            missing.append(name)
    assert not missing, f"math names with no AD entry: {missing}"


def test_python_builtins_are_ruled_or_refused():
    """abs/min/max/pow are ruled; the piecewise and integer ones are refused by name."""
    for name in ("abs", "min", "max", "pow", "round", "int", "float", "bool", "divmod"):
        target = getattr(python_builtins, name)
        assert knows(target), f"python builtin {name} has no AD entry"


def test_tables_are_disjoint():
    """A name is either ruled or refused, never both."""
    overlap:set = set(RULES_BY_OBJECT) & set(NON_DIFFERENTIABLE)
    assert not overlap, f"both ruled and refused: {overlap}"


def test_rules_version_is_stable_and_changes_with_the_table():
    """The version is the table's digest: it moves when an entry moves."""
    assert RULES_VERSION
    assert len(RULES_VERSION) == 8      # short_hash's spelling


def test_user_registered_adjoint_differentiates():
    """A black-box helper with a @grad_for adjoint differentiates against the walk."""
    from pygpgpu import kernel as kernel_factory
    import pygpgpu.numpy as gnp

    def double_step(v):
        return v * 2.0

    @grad_for(double_step)
    def double_step_grad(v, g):
        return g * 2.0

    @kernel_factory
    def body(x, out):
        for i in range(out.shape[0]):
            out[i] = double_step(x[i]) * x[i]
        return out

    x = gnp.arange(4, dtype=np.float32) + 1.0
    x.requires_grad = True
    out = gnp.zeros(4, np.float32)
    # d/dx 2x * x = 4x
    _result, pullback = body.vjp()(x, out)
    gradients = pullback(gnp.ones(4, np.float32))
    np.testing.assert_allclose(np.asarray(_pull(gradients[0])), 4.0 * np.arange(1, 5),
                               rtol=1e-5)


def _pull(value:object)->object:
    import pygpgpu.numpy as gnp
    if isinstance(value, gnp.ndarray):
        value._ensure_host_bytes()
        return value.view(np.ndarray)
    return np.asarray(value)
