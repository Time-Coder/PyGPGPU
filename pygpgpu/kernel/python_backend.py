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
  domain comes from the ``gpu.work_size(x)`` declaration (all of ``x``'s dimensions, as
  ``np.ndindex(x.shape)`` would give) or from a ``(global, local)`` bracket.

Inside a gid-style body, ``gpu.*`` resolves to host implementations installed in a
thread-local frame for the duration of the call: the indexing builtins report the
work item the serial loop is currently standing in, atomics are plain read-modify-writes
(serial execution makes them exact), ``gpu.work_size`` is a no-op, and the vector math
family delegates to the host-side :mod:`pygpgpu.vec_types.funcs`. Things that are
meaningful only with real parallel work items -- local (shared) memory -- raise.

``submit`` and ``async_call`` run the launch on a :class:`ThreadPoolExecutor` and return
a ``concurrent.futures.Future`` / ``asyncio.Future`` respectively. One launch is still
serial; concurrent launches are separate threads, so they must not share output arrays.
"""
from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .builtins import (
    ATOMIC_BUILTINS,
    INDEX_BUILTINS,
    gpu,
    get_host_frame,
    set_host_frame,
)
from .translator import _parse_kernel_source, get_dedented_source
from ..utils import local_size_for_global
from ..vec_types import funcs as VEC_FUNCS


# ---------------------------------------------------------------------------
# Source analysis
#
# The python backend runs the function body as-is, so the only thing it needs to know
# is which of the two arrangements above the body uses: does it carry the magic loop
# (run once), or does it dispatch on gpu.get_global_id (run per work item), and which
# array declares the domain with gpu.work_size(...)?
#
# Unlike the GPU translator this analysis accepts everything else in the body: it
# never inspects statements or expressions other than those three questions.
# ---------------------------------------------------------------------------

# A holder is ("arg", <parameter name>) or ("self", <attribute name>): the python
# backend never rewrites self attributes into parameters, it just reads them.
Holder = Tuple[str, str]

_MAGIC_RANGE = "range"
_MAGIC_ENUMERATE = "enumerate"
_MAGIC_NDINDEX = "ndindex"
_MAGIC_NDENUMERATE = "ndenumerate"


def _dedented_ast(func: Callable) -> Optional[ast.FunctionDef]:
    """AST of the decorated function, or None when its source is unavailable.

    ``inspect.getsource`` fails for code defined through ``exec``/stdin; such a kernel
    can still run as plain serial Python, but the launch shape cannot be derived, so the
    analysis is skipped and the frame reports that when a builtin needs it.
    """
    try:
        tree = _parse_kernel_source(get_dedented_source(func))
    except Exception:
        return None
    if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
        return None
    return tree.body[0]


def _globals_of(func: Callable) -> Dict[str, Any]:
    return getattr(func, "__globals__", None) or {}


def _holder_of(node: ast.expr, self_param: str) -> Optional[Holder]:
    """Resolve a plain parameter reference or a self.<attr> reference to a holder."""
    if isinstance(node, ast.Name):
        return "arg", node.id
    if (self_param and isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == self_param):
        return "self", node.attr
    return None


def _shape0_holder(iter_node: ast.expr, self_param: str) -> Optional[Holder]:
    """``range(<holder>.shape[0])`` -> the holder (the 1-D magic loop)."""
    if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
            and iter_node.func.id == "range" and len(iter_node.args) == 1
            and not iter_node.keywords):
        return None
    arg: ast.expr = iter_node.args[0]
    if not (isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Constant)
            and arg.slice.value == 0):
        return None
    base = arg.value
    if not (isinstance(base, ast.Attribute) and base.attr == "shape"):
        return None
    return _holder_of(base.value, self_param)


def _enumerate_holder(iter_node: ast.expr, self_param: str) -> Optional[Holder]:
    """``enumerate(<holder>)`` -> the holder (index + value snapshot magic loop)."""
    if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
            and iter_node.func.id == "enumerate" and len(iter_node.args) == 1
            and not iter_node.keywords):
        return None
    return _holder_of(iter_node.args[0], self_param)


def _nd_magic(iter_node: ast.expr, self_param: str, globals_of: Dict[str, Any]) -> Optional[Tuple[str, Holder]]:
    """``np.ndindex(<holder>.shape)`` / ``np.ndenumerate(<holder>)`` -> (form, holder)."""
    if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Attribute)
            and isinstance(iter_node.func.value, ast.Name) and len(iter_node.args) == 1
            and not iter_node.keywords):
        return None
    if globals_of.get(iter_node.func.value.id) is not np:
        return None
    form: str = iter_node.func.attr
    arg: ast.expr = iter_node.args[0]
    if form == _MAGIC_NDINDEX:
        if not (isinstance(arg, ast.Attribute) and arg.attr == "shape"):
            return None
        holder = _holder_of(arg.value, self_param)
    elif form == _MAGIC_NDENUMERATE:
        holder = _holder_of(arg, self_param)
    else:
        return None
    return (form, holder) if holder is not None else None


def _gpu_attr(call: ast.Call, globals_of: Dict[str, Any]) -> Optional[str]:
    """The ``xxx`` of a ``<gpu object>.xxx(...)`` call, or None for any other call."""
    func = call.func
    if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
        return None
    if globals_of.get(func.value.id) is not gpu:
        return None
    return func.attr


class PythonAnalysis:
    """What the serial runner needs to know about a kernel body."""

    def __init__(self) -> None:
        # Arrays declared by gpu.work_size(...): the gid-style iteration domain.
        self.declared: List[Holder] = []
        # Work size derivable from the magic loop (reported by
        # gpu.get_global_size/get_local_size/get_num_groups).
        self.magic_form: Optional[str] = None
        self.magic_holder: Optional[Holder] = None
        # The body calls a gpu indexing builtin, so it dispatches on the work item.
        self.uses_index: bool = False


def analyse(func: Callable, self_param: str = "") -> PythonAnalysis:
    """Inspect a kernel body for the two launch styles (never translates anything)."""
    analysis = PythonAnalysis()
    body: Optional[ast.FunctionDef] = _dedented_ast(func)
    if body is None:
        return analysis
    globals_of: Dict[str, Any] = _globals_of(func)

    for stmt in body.body:
        if not isinstance(stmt, ast.For):
            continue
        holder = _shape0_holder(stmt.iter, self_param)
        form = _MAGIC_RANGE
        if holder is None:
            holder = _enumerate_holder(stmt.iter, self_param)
            form = _MAGIC_ENUMERATE
        if holder is None:
            nd = _nd_magic(stmt.iter, self_param, globals_of)
            if nd is not None:
                form, holder = nd
        if holder is not None:
            analysis.magic_form = form
            analysis.magic_holder = holder
            break

    for node in ast.walk(body):
        if not isinstance(node, ast.Call):
            continue
        attr: Optional[str] = _gpu_attr(node, globals_of)
        if attr is None:
            continue
        if attr in INDEX_BUILTINS:
            analysis.uses_index = True
        elif attr == "work_size" and node.args and not node.keywords:
            holder = _holder_of(node.args[0], self_param)
            if holder is not None and holder not in analysis.declared:
                analysis.declared.append(holder)

    return analysis


# ---------------------------------------------------------------------------
# Vector-valued arrays
#
# A vector array (`np.zeros(n, dtype=float4.dtype)`) reads each element back as a
# numpy.void record, so `v.x` and `v.xyzy * 2.0` -- both fine on the GPU backends, which
# hand the work item a vector value -- fail in the interpreter. Reading through a
# VectorArray view rebuilds the vector instance from the record, and writing one
# stores its components, so the same body runs unchanged.
# ---------------------------------------------------------------------------

_VECTOR_TYPES: Dict[Any, Any] = {}


def _vector_registry() -> Dict[Any, Any]:
    """numpy dtype of a vector type -> the vector class (built once, lazily)."""
    global _VECTOR_TYPES
    if _VECTOR_TYPES:
        return _VECTOR_TYPES

    def is_vector_type(value: Any) -> bool:
        return (isinstance(value, type) and hasattr(value, "_fields_")
                and hasattr(value, "numpy_dtype") and callable(value.numpy_dtype))

    # The 2/4-wide float/int/double/uint types come from pygf and are re-exported by
    # pygpgpu.vec_types, the wider and integer-narrow ones are PyGPGPU's own: scan both.
    import pygf
    from .. import vec_types

    registry: Dict[Any, Any] = {}
    for module in (vec_types, pygf):
        for name in dir(module):
            value: Any = getattr(module, name)
            if not is_vector_type(value):
                continue
            try:
                dtype = np.dtype(value._fields_)
            except Exception:
                continue
            registry.setdefault(dtype, value)
    _VECTOR_TYPES = registry
    return registry


class VectorArray(np.ndarray):
    """Array view whose vector elements read back as vector instances instead of records."""

    def __getitem__(self, item: Any) -> Any:
        value: Any = super().__getitem__(item)
        if isinstance(value, np.void):
            vector = _vector_registry().get(value.dtype)
            if vector is not None:
                return vector.from_buffer_copy(value.tobytes())
        return value

    def __setitem__(self, item: Any, value: Any) -> None:
        if hasattr(value, "_fields_") and hasattr(value, "__len__"):
            value = tuple(value)
        super().__setitem__(item, value)


def as_host_array(value: Any) -> Any:
    """Wrap a vector-dtype array in a VectorArray view; other values pass through."""
    if not isinstance(value, np.ndarray):
        return value
    dtype = getattr(value, "dtype", None)
    if dtype is None or dtype.fields is None:
        return value
    if dtype not in _vector_registry():
        return value
    return value.view(VectorArray)


# ---------------------------------------------------------------------------
# Per-call builtin state
# ---------------------------------------------------------------------------

class PythonFrame:
    """The gpu.* host implementations of one launch, installed per thread.

    ``GPUNamespace.__getattr__`` consults the frame installed on the calling thread, so
    two concurrent launches never see each other's work item.
    """

    def __init__(self, global_size: Tuple[int, ...] = (),
                 local_size: Tuple[int, ...] = ()) -> None:
        self.global_size: Tuple[int, ...] = tuple(int(g) for g in global_size)
        self.local_size: Tuple[int, ...] = tuple(int(l) for l in local_size)
        # Global id per dimension of the work item being run; None for the plain call of
        # a magic-loop body, where there is no current work item.
        self.coords: Optional[Tuple[int, ...]] = None

    # ----- gpu.* entry point -----

    def builtin(self, name: str) -> Callable:
        """Resolve ``gpu.<name>`` for this launch."""
        if name in INDEX_BUILTINS:
            return lambda dim: self._index(name, dim)
        if name in ATOMIC_BUILTINS:
            return lambda array, index, value: self._atomic(name, array, index, value)
        if name == "barrier":
            return self._barrier
        if name == "local":
            return self._local
        if name == "work_size":
            return self._work_size
        if hasattr(VEC_FUNCS, name):
            # The dual-sided library: its host implementation is the same function.
            return getattr(VEC_FUNCS, name)
        raise SyntaxError(
            f"gpu.{name} is not supported by the python backend: it has no host-side "
            f"implementation (supported: the indexing builtins, atomics, gpu.barrier, "
            f"gpu.work_size and the vector math functions)") from None

    # ----- indexing builtins -----

    def _dim(self, name: str, dim: Any) -> int:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim not in (0, 1, 2):
            raise SyntaxError(f"gpu.{name}(dim) requires a dimension of 0, 1 or 2") from None
        return dim

    def _index(self, name: str, dim: Any) -> int:
        d: int = self._dim(name, dim)
        needs_item: bool = name in ("get_global_id", "get_local_id", "get_group_id")
        if needs_item and self.coords is None:
            raise SyntaxError(
                f"gpu.{name}({d}) has no meaning in a serial python run: the python "
                f"backend executes a magic-loop kernel as one plain call, so there is no "
                f"current work item. Use the loop variable instead (the magic loop gives "
                f"the same index).") from None
        if self.coords is not None and d >= len(self.coords):
            raise ValueError(
                f"gpu.{name}({d}) is out of range: this launch has {len(self.coords)} "
                f"work-size dimension(s)")
        if needs_item:
            assert self.coords is not None
            gid: int = self.coords[d]
            if name == "get_global_id":
                return gid
            local: int = self.local_size[d] if d < len(self.local_size) else 1
            return gid % local if name == "get_local_id" else gid // local

        if d >= len(self.global_size):
            raise ValueError(
                f"gpu.{name}({d}) is out of range: this launch has "
                f"{len(self.global_size)} work-size dimension(s); declare the domain with "
                f"gpu.work_size(array) or pass a (global, local) bracket")

        global_extent: int = self.global_size[d]
        local_extent: int = self.local_size[d] if d < len(self.local_size) else 1
        if name == "get_global_size":
            return global_extent
        if name == "get_local_size":
            return local_extent
        return -(-global_extent // local_extent)  # get_num_groups

    # ----- atomics -----

    def _atomic(self, name: str, array: Any, index: Any, value: Any) -> Any:
        """Serial atomic: one work item at a time, so a plain read-modify-write is exact."""
        if name == "atomic_xchg":
            previous = array[index]
            array[index] = value
            return previous
        current = array[index]
        if name == "atomic_add":
            array[index] = current + value
        elif name == "atomic_sub":
            array[index] = current - value
        elif name == "atomic_min":
            array[index] = min(current, value)
        elif name == "atomic_max":
            array[index] = max(current, value)
        elif name == "atomic_and":
            array[index] = current & value
        elif name == "atomic_or":
            array[index] = current | value
        elif name == "atomic_xor":
            array[index] = current ^ value
        return array[index]

    # ----- the rest of the namespace -----

    def _barrier(self) -> None:
        """Serial execution has a single work item at a time, so nothing to wait for."""
        return None

    def _local(self, *args: Any) -> Any:
        raise SyntaxError(
            "gpu.local() is not supported by the python backend: local (shared) memory is "
            "shared by the work items of one group, which a serial run cannot emulate") from None

    def _work_size(self, *args: Any) -> None:
        """Host-side declaration only: the launch shape is derived by the caller."""
        return None


# ---------------------------------------------------------------------------
# Thread pool and futures
# ---------------------------------------------------------------------------

_POOL_LOCK = threading.Lock()
_POOL: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _pool() -> concurrent.futures.ThreadPoolExecutor:
    """The shared executor behind submit()/async_call() (created on first use)."""
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = concurrent.futures.ThreadPoolExecutor(
                    thread_name_prefix="pygpgpu-python")
    return _POOL


def shutdown_thread_pool(wait: bool = True) -> None:
    """Shut the shared executor down (it is recreated on the next submit/async_call)."""
    global _POOL
    with _POOL_LOCK:
        pool: Optional[concurrent.futures.ThreadPoolExecutor] = _POOL
        _POOL = None
    if pool is not None:
        pool.shutdown(wait=wait)


def _current_loop() -> asyncio.AbstractEventLoop:
    """The running loop, else the thread's current loop, else a fresh installed one.

    ``async_call`` cannot be awaited outside a loop anyway, but returning a future
    bound to an installed loop still lets a caller drive it with
    ``loop.run_until_complete(future)`` instead of failing at creation time.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass
    loop: Optional[asyncio.AbstractEventLoop] = None
    try:
        loop = asyncio.get_event_loop()
    except Exception:
        loop = None  # no loop is set on this thread (or reading it is no longer allowed)
    if loop is not None and not loop.is_closed():
        return loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def _asyncio_future(work: concurrent.futures.Future) -> asyncio.Future:
    """Bridge a pool future to an asyncio.Future, resolving it on the event loop thread."""
    loop: asyncio.AbstractEventLoop = _current_loop()
    future: asyncio.Future = loop.create_future()

    def complete(done: concurrent.futures.Future) -> None:
        error: Optional[BaseException] = None
        try:
            error = done.exception()
        except BaseException as exc:  # cancelled or failed to deliver a result
            error = exc

        def settle() -> None:
            if future.done():
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(None)

        try:
            loop.call_soon_threadsafe(settle)
        except RuntimeError:
            pass  # the loop was closed before the launch finished

    work.add_done_callback(complete)
    return future


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

class PythonKernel:
    """Execution engine of one ``@kernel`` function on the python backend.

    Mirrors the interface of a compiled backend kernel closely enough for
    :meth:`PyKernel._invoke` to treat it as one: ``call`` / ``submit`` / ``async_call``
    with an optional ``(global, local)`` work size.
    """

    def __init__(self, func: Callable, is_method: bool, self_param: str = "") -> None:
        self.func: Callable = func
        self.is_method: bool = is_method
        self.self_param: str = self_param
        code: Any = func.__code__
        names: List[str] = list(code.co_varnames[:code.co_argcount])
        if is_method and names and names[0] == self_param:
            names = names[1:]
        self.param_names: List[str] = names
        self.analysis: PythonAnalysis = analyse(func, self_param)

    # ----- argument lookup -----

    def _resolve(self, holder: Holder, instance: Any, args: Tuple[Any, ...]) -> Any:
        kind, name = holder
        if kind == "self":
            if instance is None:
                raise ValueError(
                    f"the kernel declares gpu.work_size(self.{name}), but no instance "
                    f"was passed")
            return getattr(instance, name)
        try:
            index: int = self.param_names.index(name)
        except ValueError:
            raise ValueError(f"'{name}' is not a parameter of {self.func.__name__}()") from None
        if index >= len(args):
            raise TypeError(
                f"{self.func.__name__}() missing the argument '{name}' declared by "
                f"gpu.work_size()")
        return args[index]

    def _extents(self, holders: List[Holder], instance: Any,
                 args: Tuple[Any, ...]) -> Tuple[int, ...]:
        extents: List[int] = []
        for holder in holders:
            array = self._resolve(holder, instance, args)
            shape = getattr(array, "shape", None)
            if shape is None:
                raise ValueError(
                    f"gpu.work_size({holder[1]}) requires an array; the argument has no shape")
            extents.extend(int(s) for s in shape)
        return tuple(extents)

    # ----- caller-facing entry points -----

    def call(self, instance: Any, args: Tuple[Any, ...],
             global_size: Optional[Tuple[int, ...]] = None,
             local_size: Optional[Tuple[int, ...]] = None,
             launch: str = "call") -> Any:
        """Run (or enqueue) one launch, honouring the bracket work size if present."""
        work: Callable[[], Any] = functools.partial(
            self._run, instance, args, global_size, local_size)
        if launch == "call":
            return work()
        if launch == "submit":
            return _pool().submit(work)
        if launch == "async":
            return _asyncio_future(_pool().submit(work))
        raise ValueError(f"unknown launch style '{launch}'")

    # ----- execution -----

    def _run(self, instance: Any, args: Tuple[Any, ...],
             global_size: Optional[Tuple[int, ...]],
             local_size: Optional[Tuple[int, ...]]) -> Any:
        analysis: PythonAnalysis = self.analysis

        if analysis.magic_form is not None:
            # The loop is the launch: serial execution covers the array's real extent, so
            # a bracket cannot add or drop work items here (on a GPU it only decides how
            # the same range is split into work items). The work sizes still answer the
            # gpu.get_*size builtins.
            extents, local = self._serial_sizes(analysis, instance, args)
            return self._execute(instance, args, PythonFrame(extents, local))

        if analysis.declared:
            # A gid-style body that declared its domain with gpu.work_size(...): the
            # serial replacement of "one work item per element" is a loop over that
            # domain, each step running the body with gpu.get_global_id reporting the
            # current index. A bracket overrides the declared size, exactly as it does
            # on the GPU backends (padding the launch past the declared extent is then
            # up to the kernel's own clamp).
            if global_size is not None:
                sizes: Tuple[Tuple[int, ...], Tuple[int, ...]] = self._checked_sizes(
                    global_size, local_size)
            else:
                domain: Tuple[int, ...] = self._extents(analysis.declared, instance, args)
                if not domain:
                    raise ValueError(
                        f"gpu.work_size() declares an empty iteration domain for "
                        f"{self.func.__name__}()")
                sizes = (domain, local_size if local_size is not None
                         else local_size_for_global(domain))
            return self._run_items(instance, args, sizes[0], sizes[1])

        if global_size is not None:
            # No declaration, but the bracket states the work size: one body execution per
            # work item of that size, which is what the bracket means on every backend.
            return self._run_items(instance, args, tuple(int(g) for g in global_size),
                                   local_size)

        if analysis.uses_index:
            raise ValueError(
                f"{self.func.__name__}() dispatches on gpu.get_global_id but declares no "
                f"iteration domain, so the work size cannot be derived. Either declare "
                f"`gpu.work_size(x)` in the body or pass a bracket: "
                f"{self.func.__name__}[(total, local)](...)")

        # No magic loop, no declaration, no dispatch: the body has no notion of a work
        # item, so one plain call is the whole kernel.
        return self._execute(instance, args, PythonFrame())

    def _run_items(self, instance: Any, args: Tuple[Any, ...],
                   domain: Tuple[int, ...],
                   local: Optional[Tuple[int, ...]]) -> None:
        """Run the body once per work item of the domain."""
        if local is not None:
            local = tuple(int(l) for l in local)
            if len(local) != len(domain):
                raise ValueError(
                    f"global {domain} and local {local} work sizes must have the same dimension")
            for g, l in zip(domain, local):
                if g % l != 0:
                    raise ValueError(
                        f"global work size {domain} must be divisible by local {local}; "
                        f"use None for an automatically derived local size")
        frame = PythonFrame(domain, local or local_size_for_global(domain))
        previous: Optional[PythonFrame] = get_host_frame()
        set_host_frame(frame)
        try:
            for coords in np.ndindex(*domain):
                frame.coords = tuple(int(c) for c in coords)
                self.func(*self._call_args(instance, args))
        finally:
            set_host_frame(previous)
        return None

    def _execute(self, instance: Any, args: Tuple[Any, ...], frame: PythonFrame) -> Any:
        previous: Optional[PythonFrame] = get_host_frame()
        set_host_frame(frame)
        try:
            return self.func(*self._call_args(instance, args))
        finally:
            set_host_frame(previous)

    def _call_args(self, instance: Any, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """The positional arguments of one body execution.

        Every value is passed through :func:`as_host_array`, so a vector-dtype array
        yields vector instances element-wise; self attributes stay on the instance,
        because the body reads them through it.
        """
        call_args: Tuple[Any, ...] = tuple(as_host_array(arg) for arg in args)
        if self.is_method:
            return (instance, ) + call_args
        return call_args

    def _serial_sizes(self, analysis: PythonAnalysis, instance: Any,
                      args: Tuple[Any, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Work sizes a magic-loop kernel would have been launched with.

        Only used to answer gpu.get_global_size/get_local_size/get_num_groups; the loop
        itself runs over the real extents, so these are informational.
        """
        if analysis.magic_holder is None:
            return (), ()
        try:
            array = self._resolve(analysis.magic_holder, instance, args)
            if analysis.magic_form in (_MAGIC_NDINDEX, _MAGIC_NDENUMERATE):
                # Dimension 0 is the slowest, exactly as the GPU launch order is.
                extents = tuple(int(s) for s in reversed(tuple(array.shape)))
            elif analysis.magic_form == _MAGIC_ENUMERATE:
                extents = (len(array), )
            else:
                extents = (int(array.shape[0]), )
        except Exception:
            return (), ()
        if not extents:
            return (), ()
        return extents, local_size_for_global(extents)

    def _checked_sizes(self, global_size: Tuple[int, ...],
                       local_size: Optional[Tuple[int, ...]]
                       ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Validate a bracketed work size, exactly as the GPU backends do.

        The bracket wins over the gpu.work_size() declaration (that is what a bracket
        does on every backend): padding the global size past the declared extent is
        allowed, and the body has to clamp itself against the real element count -- which
        is what a gid-style kernel must do on a GPU too.
        """
        global_size = tuple(int(g) for g in global_size)
        if not global_size:
            raise ValueError("the global work size cannot be empty")
        if local_size is None:
            return global_size, local_size_for_global(global_size)
        local: Tuple[int, ...] = tuple(int(l) for l in local_size)
        if len(local) != len(global_size):
            raise ValueError(
                f"global {global_size} and local {local} work sizes must have the same dimension")
        for g, l in zip(global_size, local):
            if g % l != 0:
                raise ValueError(
                    f"global work size {global_size} must be divisible by local {local}; "
                    f"use None for an automatically derived local size")
        return global_size, local

    def __repr__(self) -> str:
        return f"PythonKernel({self.func.__name__})"
