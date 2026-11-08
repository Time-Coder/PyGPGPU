"""The ``python`` backend: run a ``@kernel`` function in the Python interpreter.

This backend translates nothing. It exists as the always-available fallback (and as a
reference implementation of the launch semantics): the decorated function is executed
by CPython, serially, on the host CPU, so any Python the interpreter accepts runs. It is
selected with the usual bracket, ``k["python"](...)``, and it needs no GPU runtime and
no device.

The two kernel styles need different treatment, because they are two different
arrangements of the same work-item semantics:

* a **magic loop** kernel (``for i in range(x.shape[0])``, ``enumerate(x)``,
  ``np.ndindex``/``np.ndenumerate``) already contains the loop over the elements, so
  running the function once, as plain Python, is exactly the whole kernel;
* a **gid-style** kernel (``gpu.get_global_id``) dispatches on the index of the current
  work item instead, so the body is run once per work item of the iteration domain. The
  domain comes from the ``@kernel(domain="x")`` declaration (all of ``x``'s dimensions, as
  ``np.ndindex(x.shape)`` would give) or from a ``(global, local)`` bracket.

Inside a gid-style body, ``gpu.*`` resolves to host implementations installed in a
thread-local frame for the duration of the call: the indexing builtins report the
work item the serial loop is currently standing in, atomics are plain read-modify-writes
(serial execution makes them exact), and the vector math
family delegates to the host-side :mod:`pygpgpu.vec_types.funcs`. Things that are
meaningful only with real parallel work items -- local (shared) memory -- raise.

``submit`` and ``async_call`` run the launch on a :class:`ThreadPoolExecutor` and return
a ``concurrent.futures.Future`` / ``asyncio.Future`` respectively. One launch is still
serial; concurrent launches are separate threads, so they must not share output arrays."""
# This module is the facade of a split: every name it used to define is re-exported
# from the part module that now owns it, so `python_backend.<name>` and
# `from pygpgpu.kernel.python_backend import <name>` keep working unchanged.
#
# The interpreter backend translates nothing: it runs the body as written, and only
# the launch-frame plumbing here knows what a work item is.
from __future__ import annotations


from .magic import (
    Holder,
    _MAGIC_ENUMERATE,
    _MAGIC_NDENUMERATE,
    _MAGIC_NDINDEX,
    _MAGIC_RANGE,
    _NO_RESULT,
    _dedented_ast,
    _enumerate_holder,
    _globals_of,
    _gpu_attr,
    _holder_of,
    _nd_magic,
    _shape0_holder,
)
from .analysis import (
    PythonAnalysis,
    analyse,
    as_host_array,
    prepare_structure_classes,
)
from .frame import (
    PythonFrame,
    _POOL,
    _POOL_LOCK,
    _asyncio_future,
    _current_loop,
    _pool,
    shutdown_thread_pool,
)
from .kernel import (
    PythonKernel,
)
