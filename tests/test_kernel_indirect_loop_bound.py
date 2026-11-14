"""Regression: a magic loop whose bound is a LOCAL VARIABLE multiplies the
launch grid by the array's tail (element axis), so read-modify-write stores
race. `range(img.shape[0])` directly is correct; `n = img.shape[0];
for p in range(n)` is not. Both must produce the same launch.
"""
import numpy as np
import pytest

from pygpgpu import kernel


@kernel
def indirect_bound(img: np.ndarray, amount: float) -> None:
    n = img.shape[0]
    for p in range(n):
        img[p] = img[p] + 0.014 * amount


@kernel
def direct_bound(img: np.ndarray, amount: float) -> None:
    for p in range(img.shape[0]):
        img[p] = img[p] + 0.014 * amount


@pytest.mark.parametrize("size", [64, 4096, 2211840])
def test_indirect_and_direct_bounds_agree(size):
    a = np.zeros(size, dtype=np.float32)
    indirect_bound(a, 1.0)
    expected = 0.014
    assert np.allclose(a, expected), f"indirect bound wrong at size={size}: {a.min()}..{a.max()}"

    b = np.zeros(size, dtype=np.float32)
    direct_bound(b, 1.0)
    assert np.allclose(b, expected), f"direct bound wrong at size={size}: {b.min()}..{b.max()}"

    assert np.allclose(a, b)
