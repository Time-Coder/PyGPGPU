"""The geometric vector builtins are device-function calls, not inline expansions.

The first vector-helper pass (`b60efd4`) converted the math builtins, the clamp
family, arithmetic and comparison, but left the geometric family -- `dot`,
`length`, `distance`, `normalize`, `cross` -- and the vector unary minus spelling
their formulas per component at the call site. The ray-tracing example's
`rand_unit_vector3f` therefore still read `sqrtf((result.x * result.x) + ...)`
and `-unit_vec.x` in its generated file, while the very next line showed the
conversion working (`result / len_result` emitted `vec_div_float3_float(...)`).

These tests pin the converted shape: every geometric call and every vector
negation is one `vec_<name>_<types>` device function, its definition travels in
the `device_helpers` channel, and the argument expression appears exactly once.
OpenCL keeps its native overloads and emits none of the helpers.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import (
    cross,
    dot,
    distance,
    length,
    normalize,
)
from pygpgpu import kernel
from pygf import float3
from pygpgpu.kernel.translator import FunctionTranslator
from pygpgpu.kernel.decorator import infer_ctype


def render(func, *types, backend: str) -> tuple:
    """The (kernel body, helper definitions) a translation of `func` carries.

    The resolver reads the kernel's own globals, exactly as the launch does: a builtin
    is recognized by the *object* the name resolves to, and the objects live in this
    module's imports.
    """
    raw = getattr(func, "raw_func", func)
    argument_types = [infer_ctype(t) for t in types]
    translator = FunctionTranslator(raw, argument_types, backend,
                                    lambda name: raw.__globals__.get(name), {},
                                    is_kernel=True)
    translator.translate()
    body = "\n".join(source for _, source in translator.render_kernels("k_probe"))
    return body, "\n".join(translator.device_helpers)


@kernel
def geometric_body(v: float3, w: float3, out: np.ndarray):
    """Every geometric builtin on one vector type, over expression arguments."""
    for i in range(1):
        out[0] = (dot(v, w) + length(v) + distance(v * 2.0, w) + normalize(v).x
                  + cross(v, w).y)


@kernel
def negation_body(v: float3, out: np.ndarray):
    """The unary minus on a whole vector, twice with the same operand."""
    for i in range(1):
        negated = -v
        out[0] = negated.x + (-v).y


def test_geometric_builtins_call_device_functions_on_cuda():
    """Each builtin is one `vec_<name>_<types>` call; no formula sits in the body."""
    body, helpers = render(geometric_body, float3(0.0, 0.0, 0.0),
                           float3(0.0, 0.0, 0.0), np.zeros(1, dtype=np.float32),
                           backend="cuda")
    for call in ("vec_dot_float3_float3(", "vec_length_float3(",
                 "vec_distance_float3_float3(", "vec_normalize_float3(",
                 "vec_cross_float3_float3("):
        assert call in body, call
    assert "sqrtf(" not in body
    assert "rsqrtf(" not in body
    # The helper definitions travel with the translation, C's callee-before-caller.
    for head in ("float vec_dot_float3_float3(float3 a, float3 b)",
                 "float vec_length_float3(float3 a)",
                 "float vec_distance_float3_float3(float3 a, float3 b)",
                 "float3 vec_normalize_float3(float3 a)",
                 "float3 vec_cross_float3_float3(float3 a, float3 b)"):
        assert head in helpers, head


def test_a_nested_argument_expression_appears_exactly_once():
    """`distance(v * 2.0, w)` copies the nested expression once, not per component.

    The nested `v * 2.0` itself is a helper call (`vec_mul_float3_float`), so the
    whole argument chain is two device calls -- the shape the per-component expansion
    compounded out of proportion.
    """
    body, _ = render(geometric_body, float3(0.0, 0.0, 0.0),
                     float3(0.0, 0.0, 0.0), np.zeros(1, dtype=np.float32),
                     backend="cuda")
    assert body.count("v.x * 2.0f") == 0  # no per-component expansion
    nested = "vec_distance_float3_float3(vec_mul_float3_float(v, 2.0f), w)"
    assert nested in body


def test_the_unary_minus_is_a_device_call_on_cuda():
    """`-v` is `vec_neg_float3(v)`, not three negated component reads."""
    body, helpers = render(negation_body, float3(0.0, 0.0, 0.0),
                           np.zeros(1, dtype=np.float32), backend="cuda")
    assert "vec_neg_float3(" in body
    assert "(-v.x)" not in body and "(-v.y)" not in body and "(-v.z)" not in body
    assert "float3 vec_neg_float3(float3 a)" in helpers


def test_opencl_keeps_the_native_spellings():
    """OpenCL has native overloads for all of this: no helper is emitted."""
    body, helpers = render(geometric_body, float3(0.0, 0.0, 0.0),
                           float3(0.0, 0.0, 0.0), np.zeros(1, dtype=np.float32),
                           backend="opencl")
    assert helpers == ""
    assert "dot(v, w)" in body and "length(v)" in body
    assert "vec_" not in body


def test_the_geometric_helpers_compute_the_textbook_values(runner):
    """The device function answers what the inline expansion answered.

    `v * 2.0` multiplies per component (6, 8, 10), so the distance runs against
    `(6, 8, 10) - (1, 0, 0)`; and the cross product's second component is
    `v.z*w.x - v.x*w.z` (5) -- the same slot the formula the helper spells answers.
    """
    out = np.zeros(1, dtype=np.float32)
    v = float3(3.0, 4.0, 5.0)
    w = float3(1.0, 0.0, 0.0)
    runner(geometric_body, v, w, out)
    expected: float = (
        3.0                                        # dot(v, w)
        + float(np.linalg.norm([3.0, 4.0, 5.0]))   # length(v)
        + float(np.linalg.norm([5.0, 8.0, 10.0]))  # distance(v * 2.0, w)
        + 3.0 / float(np.linalg.norm([3.0, 4.0, 5.0]))  # normalize(v).x
        + (5.0 * 1.0 - 3.0 * 0.0)                  # cross(v, w).y
    )
    assert float(out[0]) == pytest.approx(expected, abs=1e-4)


def test_the_negation_negates(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(negation_body, float3(1.0, 2.0, 3.0), out)
    assert float(out[0]) == pytest.approx(-1.0 + -2.0)
