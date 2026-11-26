# Inline kernel blocks: `with inline, kernel(...):` -- a `@kernel` body where it is launched.
"""A **block** is a `@kernel` body written at its launch site, with no `def` of its own.

::

    from pygpgpu import get_global_id, inline, kernel
    import numpy as np

    a = np.arange(1024, dtype=np.float32)
    b = np.arange(1024, dtype=np.float32) * 2.0
    out = np.zeros(1024, dtype=np.float32)

    with inline, kernel(default_device="cuda"):
        for i in range(out.shape[0]):
            out[i] = a[i] * 2.0 + b[i]

The body is the kernel -- the same magic loop, the same three host regions, the same copy
policy, the same generated file, the same cache, the same backends. Nothing about *what a
body means* changes; only where it is written. Four consequences are worth stating:

**The data comes from the scope, not from a call.** A `@kernel` body reads its values from
its parameters, a closure cell or a module global. A block has no parameter list, so the
names the body reads from the enclosing function are what travel to the device, in the order
the body first reads them. The caller writes nothing extra, and a module global needs no
capture at all (the rebuilt function's globals dictionary is the module's own).

**The caller's frame never runs the body.** Entering `kernel(...)` raises an internal signal;
`inline.__exit__` catches it and launches. That is also why `inline` must come **first** --
`with kernel(...), inline:` would enter the head before the catcher exists. The body is then
translated and launched, so a runtime failure inside it is reported against the *caller's*
line, exactly as a decorated kernel's failure is.

**A block is one-shot, so the head can say things a decorator cannot.** `domain=` accepts the
array itself or a shape tuple, because there is no later call to bind a name against.
`blocks_per_grid=` / `threads_per_block=` state the work size directly, in the same
``(blocks, threads)`` spelling a bracket uses.

**A `return` in the block answers through the object `inline` binds.** A `with` statement has
no value of its own, so `with inline as result, kernel(...):` puts the kernel's answer on
`result.value`.

Both scopes are refused rather than guessed at: a `nonlocal` (the rebuilt block is
module-level, so there is no enclosing scope to write), a top-level `break`/`continue` (they
would target the caller's loop, which the block never enters), a one-line body, and an
`await`/`yield`.

The modules below hold the pieces: `inline` the singleton and its `__exit__`, `spec` the
`kernel(...)` head, `site` the capture and the refusals, `rebuild` the generated `def`,
`domain` the `domain=` spellings, and `launch` the per-site cache and the `PyKernel` call.
"""
from __future__ import annotations

from .inline import INLINE, _InlineResult, inline
from .spec import _KernelSpec

__all__ = ["INLINE", "inline", "_InlineResult", "_KernelSpec"]
