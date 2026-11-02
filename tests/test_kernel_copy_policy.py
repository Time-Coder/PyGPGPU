"""The copy policy: what travels to the device, and what comes back.

Two rules, one per direction of waste:

* **A written buffer is always uploaded too.** The read-back copies the *whole* buffer into
  the caller's array, so a buffer that was never uploaded would come back with zeros in
  every element the kernel left alone.
* **A buffer the body neither reads nor writes is copied neither way.** The analysis knows
  the body, so an argument that appears nowhere in it cannot contribute or receive anything.

An explicit `write_only` qualifier belongs to the first rule, not the second: it describes
what the body may assume, not an instruction to stop copying.

The return claim/site words are **not** an exception to the first rule even though the
library owns them: `atomicCAS(&_ret_claim[0], 0, 1)` reads the word to compare it against
zero, so a slot that was never uploaded would decide the claim against whatever the device
buffer happened to hold.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import get_global_id, kernel
from pygpgpu.constants import KernelFlags


@kernel(domain="out")
def ignores_a_spare(spare: np.ndarray, out: np.ndarray):
    """`spare` appears nowhere in the body, so neither copy should happen."""
    i = get_global_id(0)
    out[i] = float(i)


@kernel
def returns_a_created_array(a: np.ndarray) -> np.ndarray:
    """A returning kernel, so the claim/site slots are part of the signature."""
    out = np.zeros_like(a)
    for i in range(out.shape[0]):
        out[i] = a[i] + 10
    return out


def _compiled(kernel_object, backend: str):
    decorator = kernel_object._kernel if hasattr(kernel_object, "_kernel") else kernel_object
    return next(iter(decorator._cache.values()))[0]


def test_an_untouched_argument_is_copied_neither_way(gpu_backend):
    """Rule 2: no upload, no read-back -- and the caller's array is untouched."""
    spare = np.full(64, -1.0, dtype=np.float32)
    out = np.zeros(64, dtype=np.float32)
    ignores_a_spare[gpu_backend][(1, 64)](spare, out)

    info = _compiled(ignores_a_spare, gpu_backend)._args["spare"]
    assert info.kernel_flags == KernelFlags(0), info.kernel_flags
    assert info.need_upload is False
    assert info.need_read_back is False
    assert (spare == -1.0).all(), "an untouched argument must not be overwritten"
    np.testing.assert_array_equal(out[:4], np.arange(4, dtype=np.float32))


def test_the_return_words_are_bidirectional(gpu_backend):
    """They decide the claim by *reading* zero, so both copies have to happen."""
    result = returns_a_created_array[gpu_backend](np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(result, np.arange(4, dtype=np.int32) + 10)

    compiled = _compiled(returns_a_created_array, gpu_backend)
    for name in ("_ret_claim", "_ret_site"):
        info = compiled._args[name]
        assert info.kernel_flags & KernelFlags.Readed, name
        assert info.need_upload is True, name
        assert info.need_read_back is True, name


def test_a_written_argument_is_still_uploaded(gpu_backend):
    """Rule 1: the read-back is whole-buffer, so a written buffer must travel first."""
    compiled = _compiled(returns_a_created_array, gpu_backend)
    info = compiled._args["out"]
    assert info.kernel_flags & KernelFlags.Writed
    assert info.need_upload is True, "a written buffer has to be uploaded"
    assert info.need_read_back is True
