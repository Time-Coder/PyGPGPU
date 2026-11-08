"""The `@kernel` decorator: the factory the package publishes."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, overload

from ._typing import _P
from .devices import _default_backend

if TYPE_CHECKING:
    from . import PyKernel


@overload
def kernel(func:Callable[_P, Any], *, backend:Optional[str]=None,
           domain:Optional[str]=None,
           int:object=None, float:object=None)->PyKernel[_P]: ...
@overload
def kernel(func:None=None, *, backend:Optional[str]=None,
           domain:Optional[str]=None,
           int:object=None, float:object=None)->Callable[[Callable[_P, Any]], PyKernel[_P]]: ...
def kernel(func:Optional[Callable[..., Any]]=None, *, backend:Optional[str]=None,
           domain:Optional[str]=None, int:object=None, float:object=None):
    """The `@kernel` decorator.

    When backend is not specified, probe in DEFAULT_BACKEND_ORDER (cuda -> hip -> opencl)
    and take the first backend that enumerates at least one device; the result is cached in-process.

    Usage::

        from pygpgpu.kernel import kernel
        import numpy as np

        @kernel
        def add(a, b, out):
            for i in range(out.shape[0]):
                out[i] = a[i] + b[i]

        add(np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) * 2, out)

    - The outermost `for i in range(<array parameter>.shape[0])` loop is the work-item loop;
      the work size is taken automatically from that array's first dimension;
    - plain python functions called by the kernel are translated into device functions (monomorphized per argument types);
    - helpers may be reached by name: a bare module-level name, `module.func(...)`, `SomeClass.static_method(...)`,
      `self.static_method(...)`, or `self.other_method(...)` for another method of the same class;
    - math.* and abs/min/max/round map to device builtins;
    - unsupported syntax raises SyntaxError.

    Both spellings are declared as overloads so the decorated name keeps the function's own
    parameters: ``@kernel`` and ``@kernel(backend=...)`` return one and the same
    ``PyKernel[...]``, and a checker reports a call's missing or extra argument against the
    author's ``def`` rather than against a ``*args``-only decorator.

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
    kernel's identity, so ``set_type_map()`` does not change a kernel that is already
    declared.
    """
    type_map:Optional[Dict[str, object]] = None
    if int is not None or float is not None:
        type_map = {}
        if int is not None:
            type_map["int"] = int
        if float is not None:
            type_map["float"] = float

    def decorator(f:Callable[..., Any])->"PyKernel[object]":
        # The facade defines the class; a top-level import here would be a cycle.
        from . import PyKernel  # noqa: PLC0415
        if backend is not None:
            return PyKernel(f, backend, explicit_backend=True, domain=domain,
                            type_map=type_map)
        # No backend named: the call-time default decides, so this snapshot is only used
        # for reporting until set_default_device() (or a bracket) has a say.
        return PyKernel(f, _default_backend(), explicit_backend=False, domain=domain,
                        type_map=type_map)

    if func is not None:
        return decorator(func)
    return decorator
