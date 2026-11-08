# The backend's own kernel object: the run loop over the work items.
from __future__ import annotations
from .analysis import (
    PythonAnalysis,
    analyse,
    as_host_array,
    prepare_structure_classes,
)
from .frame import (
    _asyncio_future,
    _pool,
    PythonFrame,
)
from .magic import (
    _MAGIC_ENUMERATE,
    _MAGIC_NDENUMERATE,
    _MAGIC_NDINDEX,
    _NO_RESULT,
    Holder,
)
import functools
from types import CodeType
from typing import Callable, List, Optional, Tuple
import numpy as np
from ..host_runtime import get_host_frame, set_host_frame
from ...utils import local_size_for_global



class PythonKernel:
    """Execution engine of one ``@kernel`` function on the python backend.

    Mirrors the interface of a compiled backend kernel closely enough for
    :meth:`PyKernel._invoke` to treat it as one: ``call`` / ``submit`` / ``async_call``
    with an optional ``(global, local)`` work size.
    """

    def __init__(self, func: Callable, is_method: bool, self_param: str = "",
                 domain: str = "") -> None:
        self.func: Callable = func
        self.is_method: bool = is_method
        self.self_param: str = self_param
        code: CodeType = func.__code__
        names: List[str] = list(code.co_varnames[:code.co_argcount])
        if is_method and names and names[0] == self_param:
            names = names[1:]
        self.param_names: List[str] = names
        self.analysis: PythonAnalysis = analyse(func, self_param, domain)

    # ----- argument lookup -----

    def _resolve(self, holder: Holder, instance: object, args: Tuple[object, ...]) -> object:
        kind, name = holder
        if kind == "self":
            if instance is None:
                raise ValueError(
                    f"the kernel declares @kernel(domain='{name}'), but no instance "
                    f"was passed")
            return getattr(instance, name)
        try:
            index: int = self.param_names.index(name)
        except ValueError:
            raise ValueError(f"'{name}' is not a parameter of {self.func.__name__}()") from None
        if index >= len(args):
            raise TypeError(
                f"{self.func.__name__}() missing the argument '{name}' declared by "
                f"@kernel(domain=...)")
        return args[index]

    def _extents(self, holders: List[Holder], instance: object,
                 args: Tuple[object, ...]) -> Tuple[int, ...]:
        extents: List[int] = []
        for holder in holders:
            array = self._resolve(holder, instance, args)
            shape = getattr(array, "shape", None)
            if shape is None:
                raise ValueError(
                    f"@kernel(domain='{holder[1]}') requires an array; the argument has no shape")
            extents.extend(int(s) for s in shape)
        return tuple(extents)

    # ----- caller-facing entry points -----

    def call(self, instance: object, args: Tuple[object, ...],
             global_size: Optional[Tuple[int, ...]] = None,
             local_size: Optional[Tuple[int, ...]] = None,
             launch: str = "call") -> object:
        """Run (or enqueue) one launch, honouring the bracket work size if present."""
        work: Callable[[], object] = functools.partial(
            self._run, instance, args, global_size, local_size)
        if launch == "call":
            return work()
        if launch == "submit":
            return _pool().submit(work)
        if launch == "async":
            return _asyncio_future(_pool().submit(work))
        raise ValueError(f"unknown launch style '{launch}'")

    # ----- execution -----

    def _run(self, instance: object, args: Tuple[object, ...],
             global_size: Optional[Tuple[int, ...]],
             local_size: Optional[Tuple[int, ...]]) -> object:
        # A body that builds a structure reads its members by name (`ray["origin"]`), which
        # the class has to answer: nothing else on this path would tell the class about it,
        # because the interpreter never inspects argument types.
        prepare_structure_classes(self.func)
        analysis: PythonAnalysis = self.analysis

        if analysis.magic_form is not None:
            # The loop is the launch: serial execution covers the array's real extent, so
            # a bracket cannot add or drop work items here (on a GPU it only decides how
            # the same range is split into work items).
            #
            # The group size still has to reach the work-item queries: `blockIdx`/`threadIdx`
            # and `get_local_id`/`get_group_id` divide by it, so a bracket splits the same
            # extents the way a device run would. Only those queries are affected -- the body
            # runs once, as written, so the loop itself is untouched.
            extents, auto_local = self._serial_sizes(analysis, instance, args)
            local: Optional[Tuple[int, ...]] = auto_local or None
            if extents and local_size is not None:
                if global_size is not None:
                    _, local = self._checked_sizes(global_size, local_size)
                else:
                    local = tuple(int(l) for l in local_size)
            elif extents and global_size is not None:
                _, local = self._checked_sizes(global_size, local_size)
            frame: PythonFrame = PythonFrame(extents, local or ())
            # The body stands for the first work item of the launch (see _flat_coords).
            frame.coords = tuple(0 for _ in extents) if extents else None
            return self._execute(instance, args, frame)

        if analysis.declared:
            # A gid-style body whose decorator declared the domain: the
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
                        f"@kernel(domain=...) declares an empty iteration domain for "
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
                f"`@kernel(domain=\"x\")` or pass a bracket: "
                f"{self.func.__name__}[(total, local)](...)")

        # No magic loop, no declaration, no dispatch: the body has no notion of a work
        # item, so one plain call is the whole kernel.
        return self._execute(instance, args, PythonFrame())

    def _run_items(self, instance: object, args: Tuple[object, ...],
                   domain: Tuple[int, ...],
                   local: Optional[Tuple[int, ...]]) -> object:
        """Run the body once per work item of the domain.

        The result is the first value a work item returned, matching the "first one
        wins" rule the GPU backends implement with an atomic claim: the serial loop
        simply stops taking results once one has been produced, so later work items
        still run (they may write their own outputs) but cannot change the result.
        """
        if local is not None:
            local = tuple(int(l) for l in local)
            if len(local) != len(domain):
                raise ValueError(
                    f"the block size {local} names {len(local)} dimension(s) but the launch "
                    f"has {len(domain)}; both have to name the same number of axes")
        frame = PythonFrame(domain, local or local_size_for_global(domain))
        previous: Optional[PythonFrame] = get_host_frame()
        set_host_frame(frame)
        winner: object = _NO_RESULT
        try:
            for coords in np.ndindex(*domain):
                frame.coords = tuple(int(c) for c in coords)
                result: object = self.func(*self._call_args(instance, args))
                if winner is _NO_RESULT and result is not None:
                    winner = result
        finally:
            set_host_frame(previous)
        return None if winner is _NO_RESULT else winner

    def _execute(self, instance: object, args: Tuple[object, ...], frame: PythonFrame) -> object:
        previous: Optional[PythonFrame] = get_host_frame()
        set_host_frame(frame)
        try:
            return self.func(*self._call_args(instance, args))
        finally:
            set_host_frame(previous)

    def _call_args(self, instance: object, args: Tuple[object, ...]) -> Tuple[object, ...]:
        """The positional arguments of one body execution.

        Every value is passed through :func:`as_host_array`, so a vector-dtype array
        yields vector instances element-wise; self attributes stay on the instance,
        because the body reads them through it.
        """
        call_args: Tuple[object, ...] = tuple(as_host_array(arg) for arg in args)
        if self.is_method:
            return (instance, ) + call_args
        return call_args

    def _serial_sizes(self, analysis: PythonAnalysis, instance: object,
                      args: Tuple[object, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
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

        The bracket wins over the `@kernel(domain=...)` declaration (that is what a bracket
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
                f"the block size {local} names {len(local)} dimension(s) but the launch has "
                f"{len(global_size)}; both have to name the same number of axes")
        return global_size, local

    def __repr__(self) -> str:
        return f"PythonKernel({self.func.__name__})"
