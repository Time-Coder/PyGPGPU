"""Boolean operators in a kernel body.

`and`/`or` used to be emitted with the operator in front of the operands --
``( &&(i < n) (j < m))`` -- which is not a C expression at all, so any kernel that combined
two conditions failed to compile. Nothing in the suite used them, which is why it survived;
the shared-memory histogram in the image example is what finally exercised it.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def both_conditions(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        if x[i] > 0.0 and x[i] < 0.5:
            out[i] = 1.0
        else:
            out[i] = 0.0


@kernel
def either_condition(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        if x[i] < 0.0 or x[i] > 1.0:
            out[i] = 1.0
        else:
            out[i] = 0.0


@kernel
def three_conditions(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        if x[i] > 0.0 and x[i] < 1.0 and x[i] != 0.5:
            out[i] = 2.0
        else:
            out[i] = 0.0


@kernel
def negated_and(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        if not (a[i] > 0.0 and b[i] > 0.0):
            out[i] = 1.0
        else:
            out[i] = 0.0


@kernel
def combined_value(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    """Using the boolean as a value, not only as a condition."""
    for i in range(out.shape[0]):
        out[i] = int(a[i] > 0.0 and b[i] > 0.0)


def test_and_in_a_condition(runner):
    x = np.array([-1.0, 0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    out = np.zeros(5, dtype=np.float32)
    runner(both_conditions, x, out)
    np.testing.assert_allclose(out, np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_or_in_a_condition(runner):
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    out = np.zeros(5, dtype=np.float32)
    runner(either_condition, x, out)
    np.testing.assert_allclose(out, np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_a_chain_of_three_conditions(runner):
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    out = np.zeros(5, dtype=np.float32)
    runner(three_conditions, x, out)
    np.testing.assert_allclose(out, np.array([0.0, 2.0, 0.0, 2.0, 0.0], dtype=np.float32))


def test_a_negated_conjunction(runner):
    a = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    b = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(negated_and, a, b, out)
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32))


def test_a_conjunction_used_as_a_value(runner):
    a = np.array([1.0, 1.0, -1.0], dtype=np.float32)
    b = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(combined_value, a, b, out)
    np.testing.assert_allclose(out, np.array([1.0, 0.0, 0.0], dtype=np.float32))
