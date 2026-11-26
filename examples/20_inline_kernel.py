"""Inline kernel blocks: `with inline, kernel(...):` -- a `@kernel` body where it is launched.

A block is the same body a `@kernel` function would carry, written at the call site instead of
inside a `def`. The body is never executed as python by this frame: entering the head raises an
internal signal, `inline.__exit__` catches it, and the block is translated and launched through
the ordinary `@kernel` pipeline. So the magic loop, the three host regions, the copy policy, the
generated file, the cache and the backends are all the same -- only where the body is written
changes.

The domain is always the magic loop's (`for i in range(out.shape[0])`). A block is written where
it is launched, so the array that gives the work size is already in scope; a gid-style body
(`get_global_id`) needs the domain named outside the body, which is `@kernel(domain=...)`'s job,
so it is refused here by name.

Run: python examples/20_inline_kernel.py
"""
import numpy as np

import pygpgpu
from pygpgpu import get_global_id, inline, kernel


def element_wise(backend:str)->None:
    """The common case: the data comes from this scope, the head carries only options."""
    a = np.arange(8, dtype=np.float32)
    b = np.arange(8, dtype=np.float32) * 2.0
    out = np.zeros(8, dtype=np.float32)

    with inline, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = a[i] * 2.0 + b[i]

    assert np.allclose(out, a * 2.0 + b), out
    print(f"  element-wise on {backend}: {out[:4]}...")


def a_captured_scalar(backend:str)->None:
    """A local of this function travels as a kernel parameter, in the order the body reads it."""
    a = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    factor = 2.5

    with inline, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = a[i] * factor

    assert np.allclose(out, a * 2.5), out
    print(f"  captured scalar on {backend}: {out}")


def two_passes(backend:str)->None:
    """Two magic loops are two device passes, exactly as in a `@kernel` body."""
    a = np.arange(4, dtype=np.float32)
    mid = np.zeros(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)

    with inline, kernel(default_device=backend):
        for i in range(mid.shape[0]):
            mid[i] = a[i] + 1.0
        for i in range(out.shape[0]):
            out[i] = mid[i] * 2.0

    assert np.allclose(mid, a + 1.0), mid
    assert np.allclose(out, (a + 1.0) * 2.0), out
    print(f"  two passes on {backend}: {out}")


def a_host_region(backend:str)->None:
    """A statement above the loop runs once on the host; the tail runs after the launch."""
    a = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    flag = np.zeros(1, dtype=np.float32)

    with inline, kernel(default_device=backend):
        width = a.shape[0]                     # host region, before the launch
        for i in range(out.shape[0]):
            out[i] = a[i] + width
        flag[0] = float(out[0])                # the tail, after it

    assert np.allclose(out, a + 4.0), out
    assert flag[0] == out[0]
    print(f"  host regions on {backend}: out={out} tail={flag}")


def working_size(backend:str)->None:
    """The work size may be stated from the head, in the bracket's own (blocks, threads) pair.

    The *domain* is not stated there: it is the magic loop's bound, which is why a block
    always carries one. The pair only says how that domain is split into work items.
    """
    a = np.arange(8, dtype=np.float32)
    out = np.zeros(8, dtype=np.float32)

    with inline, kernel(default_device=backend, blocks_per_grid=2, threads_per_block=4):
        for i in range(out.shape[0]):
            out[i] = a[i] + 1.0

    assert np.allclose(out, a + 1.0), out
    print(f"  blocks_per_grid=2, threads_per_block=4 on {backend}: {out[:4]}...")


def the_result(backend:str)->None:
    """A `with` has no value, so a kernel's return lands on the object `inline` binds."""
    out = np.zeros(4, dtype=np.float32)
    a = np.arange(4, dtype=np.float32)

    with inline as result, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = a[i]
        return 42

    assert result.value == 42, result
    print(f"  return through `as result` on {backend}: result.value={result.value}")


def one_kernel_per_site(backend:str)->None:
    """A block in a loop translates once: the site keeps its kernel, like a decorated one."""
    from pygpgpu.kernel.inline_kernels.launch import _SITES

    out = np.zeros(4, dtype=np.float32)
    a = np.arange(4, dtype=np.float32)
    before = len(_SITES)
    for _ in range(3):
        with inline, kernel(default_device=backend):
            for i in range(out.shape[0]):
                out[i] = out[i] + a[i] + 1.0
    assert len(_SITES) == before + 1, (before, len(_SITES))
    assert np.allclose(out, a * 3.0 + 3.0), out
    print(f"  three executions, one translation on {backend}")


def refusals(backend:str)->None:
    """What the spelling cannot mean is refused before anything is translated."""
    out = np.zeros(4, dtype=np.float32)
    try:
        with inline:
            out[0] = 1.0
    except SyntaxError as error:
        print(f"  inline alone -> SyntaxError: {str(error).splitlines()[0]}")

    try:
        with inline, kernel(default_device=backend, domain="out"):
            for i in range(out.shape[0]):
                out[i] = 1.0
    except SyntaxError as error:
        print(f"  domain= on a block -> SyntaxError: {str(error).splitlines()[0]}")

    try:
        with inline, kernel(default_device=backend):
            lane = get_global_id(0)
            if lane < out.shape[0]:
                out[lane] = 1.0
    except SyntaxError as error:
        print(f"  gid-style body -> SyntaxError: {str(error).splitlines()[0]}")


def main()->None:
    # The interpreter backend always works, plus one real device when this machine has any.
    available = pygpgpu.list_devices()
    backends = ["python"] + available[:1]
    print(f"backends: {backends}")
    for backend in backends:
        print(f"--- {backend} ---")
        element_wise(backend)
        a_captured_scalar(backend)
        two_passes(backend)
        a_host_region(backend)
        working_size(backend)
        the_result(backend)
        one_kernel_per_site(backend)
        refusals(backend)
    print("ok")


if __name__ == "__main__":
    main()
