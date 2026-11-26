"""The `@kernel` decorator: the factory the package publishes."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Union, overload

from ._typing import _P
from .devices import DefaultDeviceEntries, _parse_default_device_value, _snapshot_backend

if TYPE_CHECKING:
    from . import PyKernel
    from ..inline_kernels.spec import _KernelSpec


# The device setting every public spelling accepts: `default_device=` and its alias `device=`.
_DeviceValue = Union[str, Sequence[str]]


@overload
def kernel(func:Callable[_P, Any], *,
           default_device:Optional[_DeviceValue]=None,
           device:Optional[_DeviceValue]=None,
           domain:Optional[str]=None,
           int:object=None, float:object=None)->PyKernel[_P]: ...
@overload
def kernel(func:None=None, *,
           default_device:Optional[_DeviceValue]=None,
           device:Optional[_DeviceValue]=None,
           domain:Optional[str]=None,
           int:object=None, float:object=None,
           blocks_per_grid:object=None,
           threads_per_block:object=None)->_KernelSpec: ...
def kernel(func:Optional[Callable[..., Any]]=None, *,
           default_device:Optional[_DeviceValue]=None,
           device:Optional[_DeviceValue]=None,
           domain:Optional[str]=None, int:object=None, float:object=None,
           blocks_per_grid:object=None, threads_per_block:object=None):
    """The `@kernel` decorator, and the head of an inline kernel block.

    Two roles, one name, because the `with` spelling is written `kernel(...)` either way.
    Called with a function it decorates; called with options only it returns a decorator
    factory -- and that same object is what ``with inline, kernel(...):`` enters.

    Usage::

        @kernel
        def add(a, b, out):
            for i in range(out.shape[0]):
                out[i] = a[i] + b[i]

        add(np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) * 2, out)

        # the body, written where it is launched, with no `def`
        with inline, kernel(device="cuda"):
            for i in range(out.shape[0]):
                out[i] = a[i] * 2.0 + b[i]

    - The outermost `for i in range(<array>.shape[0])` loop is the work-item loop;
      the work size is taken automatically from that array's first dimension;
    - plain python functions called by the kernel are translated into device functions (monomorphized per argument types);
    - helpers may be reached by name: a bare module-level name, `module.func(...)`, `SomeClass.static_method(...)`,
      `self.static_method(...)`, or `self.other_method(...)` for another method of the same class;
    - math.* and abs/min/max/round map to device builtins;
    - unsupported syntax raises SyntaxError.

    Both decorator spellings are declared as overloads so the decorated name keeps the
    function's own parameters: ``@kernel`` and ``@kernel(default_device=...)`` return one
    and the same ``PyKernel[...]``, and a checker reports a call's missing or extra argument
    against the author's ``def`` rather than against a ``*args``-only decorator.

    `default_device=` names what this kernel's bracket-less calls run on, accepting the
    same shapes and the same selector strings `config.default_device` takes -- one
    (``"cuda:0"``, ``"opencl:cpu"``, a bare device name) or a list tried in order, each
    entry used when the entries before it are unusable::

        @kernel(default_device="cuda")                   # plain calls use cuda
        def add(a, b, out): ...

        @kernel(default_device=["opencl:cpu", "cuda"])   # ... and fall back to cuda
        def add(a, b, out): ...

    **`device=` is the same option under a shorter name**, for the call sites where it reads
    better (`with inline, kernel(device="cuda"):`). The two are one option: either spelling
    accepts the same values and means the same thing, and passing both is refused rather than
    silently letting one win.

    A kernel that names one walks its own list and ignores the process-wide setting;
    a bracket on the call (``k["opencl:1"](...)``) still overrides both. The entries are
    resolved at call time -- an unusable one is skipped, and a device attached later is
    found -- while a wrong selector is refused when the decorator runs. ``None`` (or an
    empty list) leaves the choice to `config.default_device`.

    `int=` / `float=` choose the device width a python ``int`` / ``float`` travels as, because
    neither has a width of its own::

        @kernel(float=np.float64)          # python floats and python complex values are 64-bit
        def accurate(scale, out):
            ...

        @kernel(int=np.int64)              # a python int is a 64-bit counter here
        def count(step, out):
            ...

    The value is a type object -- a numpy dtype/scalar type (``np.int64``), a ctypes scalar
    class (``ctypes.c_int64``) -- or a device type name (``"long"``). Only the two python
    builtins are mapped: a numpy scalar or array already names its own width and keeps it
    (``np.uint32(x)`` is still ``uint``). A python ``complex`` has no width either, so it
    follows ``float=``. The choice is fixed when the decorator runs and is part of the
    kernel's identity, so ``config.type_map`` does not change a kernel that is already
    declared.

    `domain=` declares the work size of a gid-style body: it names an array **parameter** of a
    decorated kernel (``@kernel(domain="out")``). An inline block does not take a `domain=`:
    it states its domain with a magic loop in the body instead, and a gid-style body is
    refused there by name (`with inline, kernel(...)` is written where it is launched, so the
    extent it iterates is already in scope).

    `blocks_per_grid=` / `threads_per_block=` state the work size from an inline block's head,
    in the same ``(blocks, threads)`` spelling a bracket uses
    (`k[(blocks, threads)](...)`), which is CUDA's pair on every backend.
    """
    # `device=` is `default_device=` under a shorter name. Both spellings name one option, so
    # giving both is ambiguous rather than layered -- refuse it instead of letting the later
    # keyword silently win.
    if device is not None:
        if default_device is not None:
            raise TypeError(
                "@kernel() takes one device setting: use `default_device=` or its alias "
                "`device=`, not both (they mean the same thing)")
        default_device = device
    type_map:Optional[Dict[str, object]] = None
    if int is not None or float is not None:
        type_map = {}
        if int is not None:
            type_map["int"] = int
        if float is not None:
            type_map["float"] = float

    # The kernel's own default-device entries, parsed where the decorator runs: a wrong
    # selector is a decoration-time refusal, not a launch-time surprise. None leaves the
    # bracket-less call to config.default_device.
    default_entries:Optional[DefaultDeviceEntries] = (
        _parse_default_device_value(default_device, "@kernel(default_device=...)") or None)

    if func is not None:
        # The bare `@kernel` spelling: decorate now.
        from . import PyKernel  # noqa: PLC0415
        # The backend argument is a reporting snapshot only: the walk runs per call, and
        # every launch records the backend it actually used.
        return PyKernel(func, _snapshot_backend(default_entries),
                        default_devices=default_entries, domain=domain,
                        type_map=type_map)

    # Called with options only. The result is a decorator factory -- and the `with`-item head
    # of an inline kernel block, which is why it is not a plain function.
    from ..inline_kernels.spec import _KernelSpec  # noqa: PLC0415

    return _KernelSpec(default_entries, domain, type_map,
                       blocks_per_grid=blocks_per_grid,
                       threads_per_block=threads_per_block)
