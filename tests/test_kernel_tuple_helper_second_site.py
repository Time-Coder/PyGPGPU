"""A tuple-returning helper called a SECOND time in one function.

The first call site of a tuple-returning helper reaches `_emit_device_call`'s
cache-miss path, which appends the unpack destinations as the helper's
`_tuple_out*` pointer arguments (the destinations travel by address -- see
`_emit_tuple_call_unpack`). A *second* call site of the same monomorphized
helper hits the device cache instead, and that path used to return without
appending them: the generated call had too few arguments and NVRTC refused the
whole kernel at its first launch ("too few arguments in function call"). The
first site looked correct, so every single-site test passed.

The trigger is "the same helper, twice, in one function" -- directly in the
kernel body, or one function removed, inside a nested device helper (the
shape the geodesic tracer that found it actually wrote:
pygpgpu_video/pygvideo/kernels/blackhole.py, whose disk_emit answers
(color, opacity) and is called once per hole per crossing). Neither renaming
the unpack targets nor annotating them changes anything, because the miss is
in the argument list, not the destinations. The workaround -- splitting the
tuple into single-value helpers, each called twice -- never travels the
cache-hit tuple path, so it kept working throughout and stays pinned here as
a regression guard.

One test renders the source without a device (the bug lived entirely in the
generated text, so the render catches it on a machine with no GPU): both call
sites must carry the destinations.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.kernel.decorator import infer_ctype
from pygpgpu.kernel.translator import FunctionTranslator


# ---------------------------------------------------------------------------
# The helpers and kernels: the three shapes the bug report carried
# ---------------------------------------------------------------------------


def emit_pair(v, k):
    """Any tuple-returning device helper: two values from two numbers."""
    return (v * k, v * v)


@kernel
def flat_two_sites(buf: np.ndarray, k: float):
    """A: two call sites directly in the kernel body."""
    for i in range(buf.shape[0]):
        a0, a1 = emit_pair(buf[i], k)
        b0, b1 = emit_pair(buf[i] + 1.0, k)
        buf[i] = a0 + a1 + b0 + b1


def trace_once(v, k):
    """A nested helper that calls the tuple helper twice."""
    a0, a1 = emit_pair(v, k)
    b0, b1 = emit_pair(v + 1.0, k)
    return a0 + a1 + b0 + b1


@kernel
def nested_two_sites(buf: np.ndarray, k: float):
    """B: kernel -> nested helper -> the tuple helper, twice."""
    for i in range(buf.shape[0]):
        buf[i] = trace_once(buf[i], k)


def emit_value(v, k):
    """The workaround's first half: one value per helper."""
    return v * k


def emit_weight(v):
    """The workaround's second half."""
    return v * v


@kernel
def split_two_sites(buf: np.ndarray, k: float):
    """C: the split-helper workaround, each half called twice."""
    for i in range(buf.shape[0]):
        v = buf[i]
        buf[i] = (emit_value(v, k) + emit_weight(v)
                  + emit_value(v + 1.0, k) + emit_weight(v + 1.0))


def reference(v: np.ndarray, k: float) -> np.ndarray:
    """What the three kernels all compute."""
    v2 = v + 1.0
    return (v * k + v * v + v2 * k + v2 * v2).astype(np.float32)


def _make_buffer() -> np.ndarray:
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buf = np.zeros(4, np.float32)
    buf[:] = values
    return buf


def test_two_tuple_call_sites_in_the_kernel_body(runner):
    """A: the second unpack must compile, not only the first."""
    buf = _make_buffer()
    runner(flat_two_sites, buf, 2.0)
    np.testing.assert_allclose(
        buf, reference(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), 2.0))


def test_two_tuple_call_sites_inside_one_nested_helper(runner):
    """B: the sites sit in `trace_once`, one translation level below the kernel."""
    buf = _make_buffer()
    runner(nested_two_sites, buf, 2.0)
    np.testing.assert_allclose(
        buf, reference(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), 2.0))


def test_the_split_helper_workaround_still_runs(runner):
    """C: one helper per value, each called twice -- never touched the bug."""
    buf = _make_buffer()
    runner(split_two_sites, buf, 2.0)
    np.testing.assert_allclose(
        buf, reference(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), 2.0))


# ---------------------------------------------------------------------------
# A raising tuple helper: the destinations come before the exception pointers
# ---------------------------------------------------------------------------


def checked_pair(v, k):
    """A tuple-returning helper that can raise.

    Its signature then ends in the tuple destinations, the exception slot and
    the stop flag, in that order (see `render_device`) -- so the call site
    forwarding has to append the destinations *before* the two pointers, on
    both the cache-miss path and the cache-hit path this file pins.
    """
    if v < 0.0:
        raise ValueError("the value must not be negative")
    return (v * k, v * v)


@kernel
def raising_tuple_two_sites(buf: np.ndarray, k: float):
    """Two call sites of a raising tuple helper, in one function."""
    for i in range(buf.shape[0]):
        a0, a1 = checked_pair(buf[i], k)
        b0, b1 = checked_pair(buf[i] + 1.0, k)
        buf[i] = a0 + a1 + b0 + b1


def test_a_raising_tuple_helper_runs_twice(gpu_backend):
    """Both sites compile and answer, with the exception pointers forwarded."""
    buf = _make_buffer()
    raising_tuple_two_sites[gpu_backend](buf, 2.0)
    np.testing.assert_allclose(
        buf, reference(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), 2.0))


def test_a_raising_tuple_helper_raises_with_its_message(gpu_backend):
    """The guard inside the helper is reachable from the second call site too."""
    buf = _make_buffer()
    buf[2] = -1.0
    with pytest.raises(ValueError, match="the value must not be negative"):
        raising_tuple_two_sites[gpu_backend](buf, 2.0)


# ---------------------------------------------------------------------------
# The render-only pin
# ---------------------------------------------------------------------------


def generate(func, *types, backend: str = "cuda") -> str:
    """The rendered device source for `func`, without compiling it.

    The resolver is the function's own module globals, so the helper calls
    resolve (a `lambda name: None` answers for kernels that call no helpers).
    The helpers' own definitions are not in `render_kernels`' output -- they
    live in the device cache as their entries' `code` -- so they are joined in
    too: the tuple signature being pinned is one of those definitions.
    """
    raw = getattr(func, "raw_func", func)
    argument_types = [infer_ctype(t) for t in types]
    translator = FunctionTranslator(raw, argument_types, backend,
                                    lambda name: raw.__globals__.get(name), {},
                                    is_kernel=True)
    translator.translate()
    pieces = [source for _, source in translator.render_kernels("k_probe")]
    pieces.extend(str(entry["code"]) for entry in translator.device_cache.values()
                  if isinstance(entry, dict) and entry.get("status") == "done"
                  and entry.get("code"))
    return "\n\n".join(pieces)


def test_the_second_call_site_passes_the_tuple_destinations():
    """Both call sites carry the helper's out pointers, in the rendered text.

    The miss was exactly this: the cache-hit path built the call without the
    destinations, so the second site read `emit_pair_(buf[i] + 1.0f), k);`
    against a signature that ends in `float* _tuple_out0, float* _tuple_out1`.
    Each destination appears once, at its own site.
    """
    source = generate(flat_two_sites, np.zeros(4, np.float32), np.float32(2.0))
    assert "_tuple_out0" in source and "_tuple_out1" in source
    assert source.count("&(a0)") == 1 and source.count("&(a1)") == 1
    assert source.count("&(b0)") == 1 and source.count("&(b1)") == 1


def test_the_raising_tuple_helper_forwards_destinations_before_the_exception():
    """Both call sites carry the destinations, then the exception pointers.

    The order is the signature's (destinations, slot, stop -- see
    `render_device`), and it has to hold on the cache-miss path (the first
    site) and the cache-hit path (the second) alike.
    """
    source = generate(raising_tuple_two_sites, np.zeros(4, np.float32),
                      np.float32(2.0))
    sites = [line for line in source.splitlines()
             if "checked_pair_" in line and ("&(a" in line or "&(b" in line)]
    assert len(sites) == 2
    for line in sites:
        assert line.index("&(") < line.index("_exception_site")
