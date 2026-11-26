# `kernel(...)` as a block head: the options, and the `with`-item half of the spelling.
"""`kernel(...)` has two roles, and this module holds the second one.

Called with a function, it is the decorator (see `decorator/factory.py`). Called as the last
item of `with inline, kernel(...):`, it is the **block head**: it carries the options and,
when entered, captures the block. The two spellings share one callable, because the `with`
line is written `kernel(...)` either way.

What the head does *not* carry is the data. A `@kernel` body reads its values from its
parameters, a closure cell or a module global; a block has no parameter list, so the values
come from the enclosing frame's scope exactly as they would for that `@kernel` body. That is
what makes the block one-shot: there is no signature to bind a later call against.

Options:

* `default_device=`, `int=`, `float=` -- the decorator's own, with the same meaning;
* `blocks_per_grid=` / `threads_per_block=` -- the work size, in the ``(blocks, threads)``
  spelling a bracket uses (`k[(blocks, threads)](...)`), which is CUDA's pair everywhere.

`domain=` is deliberately **not** here. An inline block states its domain with a magic loop
in the body (`for i in range(x.shape[0])`), because the extent it iterates is already in
scope at the launch site; a gid-style body would have to name the domain outside the body,
and `@kernel(domain=...)` is that spelling. See `site._refuse_a_gid_style_body`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

from ..decorator.devices import (
    DefaultDeviceEntries,
    _snapshot_backend,
)

if TYPE_CHECKING:
    from ..decorator import PyKernel


def _work_size(blocks_per_grid:object,
               threads_per_block:object) -> Tuple[Optional[Tuple[int, ...]],
                                                  Optional[Tuple[int, ...]]]:
    """The head's work-size pair as the ``(global, local)`` form the launch speaks.

    ``blocks_per_grid`` alone is a bare item count (the same reading a one-element bracket
    has); with ``threads_per_block`` the two are CUDA's ``(blocks, threads)`` pair, converted
    by the same `_parse_work_item` a bracket goes through, so there is one conversion rule.
    """
    from ..decorator.devices import _parse_work_item  # noqa: PLC0415

    if threads_per_block is None:
        return _parse_work_item(blocks_per_grid)
    return _parse_work_item((blocks_per_grid, threads_per_block))


class _KernelSpec:
    """The `with`-item form of `kernel(...)`: options now, a block when entered.

    It is **callable**, so `@kernel(domain="x")` keeps working unchanged: called with a
    function it decorates, exactly as the factory does. That is what lets one name serve both
    spellings instead of a second entry point.

    Declared as a context manager in its own right (the `__enter__`/`__exit__` below are the
    real protocol), so a checker accepts `with inline, kernel(...):` without a suppression and
    still sees the decorator's call signature.
    """

    def __init__(self,
                 default_devices:Optional[DefaultDeviceEntries]=None,
                 domain:Optional[str]=None,
                 type_map:Optional[Dict[str, object]]=None,
                 blocks_per_grid:object=None,
                 threads_per_block:object=None)->None:
        # A decorator's `domain=` is a parameter *name*, checked where the head is built so a
        # wrong spelling fails at the decorator -- not at the first launch, and not silently
        # as a `str()` of whatever was passed.
        if domain is not None and not isinstance(domain, str):
            raise SyntaxError(
                f"@kernel(domain=...) takes the name of an array parameter, as a string "
                f"(e.g. domain=\"out\"), because a decorated kernel is written once and "
                f"called many times; got {type(domain).__name__}. An inline kernel block "
                f"takes no domain at all -- its body states the domain with a magic loop")
        self.default_devices:Optional[DefaultDeviceEntries] = default_devices
        self.domain:Optional[str] = domain
        self.type_map:Optional[Dict[str, object]] = type_map
        self.global_size:Optional[Tuple[int, ...]]
        self.local_size:Optional[Tuple[int, ...]]
        self.global_size, self.local_size = _work_size(blocks_per_grid, threads_per_block)
        stated_size:bool = (blocks_per_grid is not None or threads_per_block is not None)
        if domain is not None and stated_size:
            raise SyntaxError(
                "@kernel(domain=...) and blocks_per_grid=/threads_per_block= both state the "
                "work size, so an inline kernel may carry one of them, not both: the domain "
                "derives the size from an array's shape, the pair states it directly")

    # ----- the decorator half -----

    def __call__(self, func:Callable[..., object])->"PyKernel":
        from ..decorator import PyKernel  # noqa: PLC0415

        return PyKernel(func, _snapshot_backend(self.default_devices),
                        default_devices=self.default_devices,
                        domain=self.domain, type_map=self.type_map)

    # ----- the block half -----

    def __enter__(self)->"_KernelSpec":
        """Capture the block and signal it: the caller's frame must not run the body."""
        from .inline import INLINE  # noqa: PLC0415
        from .site import capture  # noqa: PLC0415

        raise capture(self, INLINE)

    def __exit__(self, exc_type:object, exc:object,
                 traceback:object)->bool:
        # Unreachable through the documented spelling: `inline` is entered first, so its
        # `__exit__` catches the signal before this one unwinds. Reaching here means the
        # order was written the other way round, which is the loud failure the message names.
        raise SyntaxError(
            "`inline` must come first: write `with inline, kernel(...):`. Written the other "
            "way round (`with kernel(...), inline:`), the kernel head is entered before the "
            "catcher exists, so nothing could suppress the block body")
