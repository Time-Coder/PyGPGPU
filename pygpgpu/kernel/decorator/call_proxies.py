# The objects a bracket or a bound method returns, and the tail-return exception.
from __future__ import annotations
from typing import TYPE_CHECKING
from ._typing import _KernelBase, _P
from .binding import _bind_arguments, _native_argument_errors
from .devices import _parse_selector, _parse_work_item
if TYPE_CHECKING:
    from . import PyKernel
    # `asyncio` names annotations only (`async_call`'s return type), which are lazy
    # under `from __future__ import annotations` -- so the plain-launch import path
    # never loads it.
    import asyncio
import ast
import concurrent.futures
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)



class _KernelCall(_KernelBase[_P]):
    """Bracket-configuration form: k["cuda"][(64, 8)](args) / k[(64, 8)]["cuda"](args), any order.

    Generic over the decorated function's parameters: a bracket only chooses a backend, a
    device or a work size, so the call it finally makes has the kernel's own signature --
    which is what lets ``k["cuda"](a, out)`` suggest ``a``/``out`` just as ``k(a, out)`` does.
    """

    def __init__(self, kernel:"PyKernel[_P]", instance:object,
                 backend:Optional[str]=None, device:Optional[str]=None,
                 global_size:Optional[Tuple[int, ...]]=None, local_size:Optional[Tuple[int, ...]]=None):
        self._kernel:"PyKernel[_P]" = kernel
        self._instance:object = instance
        self._backend:Optional[str] = backend
        self._device:Optional[str] = device
        self._global_size:Optional[Tuple[int, ...]] = global_size
        self._local_size:Optional[Tuple[int, ...]] = local_size

    def __getitem__(self, item:object)->"_KernelCall[_P]":
        if isinstance(item, str):
            backend, device = _parse_selector(item)
            return _KernelCall(
                self._kernel, self._instance,
                (backend if backend is not None else self._backend),
                (device if device is not None else self._device),
                self._global_size, self._local_size)
        global_size, local_size = _parse_work_item(item)
        return _KernelCall(
            self._kernel, self._instance, self._backend, self._device,
            (global_size if global_size is not None else self._global_size),
            (local_size if local_size is not None else self._local_size))

    @_native_argument_errors
    # The result is the kernel body's value, produced on the device. `_R` is the
    # decorated function's result type but is not bound on this class, so the
    # honest annotation is the untyped one -- `object` would forbid unpacking a
    # tuple return at the call site.
    def __call__(self, *args:_P.args, **kwargs:_P.kwargs)->Any:
        instance:object = self._instance
        instance_arg:Optional[object] = None
        if self._kernel._is_classmethod or self._kernel._self_param == "cls":
            # A classmethod kernel. Through the descriptor the class is already bound
            # (`instance` carries it); through the bare PyKernel nothing binds it, and
            # the owner resolves from the kernel's own qualified name -- unless the
            # caller passes the class explicitly as the leading argument (the one
            # spelling that works for a class the qualname cannot resolve, a
            # test-local class, say). Either way the rest are arguments.
            if instance is None and args and isinstance(args[0], type):
                instance = args[0]
                args = args[1:]
            call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                        skip_first=True)
            return self._kernel._invoke(instance, call_args, self._backend, self._device,
                                    self._global_size, self._local_size)
        if self._kernel._is_method and instance is None:
            # class form: Klass.method["cuda"](instance, args...)
            if not args:
                raise TypeError(
                    f"{self._kernel.__name__}() is an instance method; pass the instance "
                    f"as the first argument")
            instance_arg = args[0]
            args = args[1:]
            instance = instance_arg
        call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=self._kernel._is_method)
        return self._kernel._invoke(
            instance, call_args, self._backend, self._device, self._global_size, self._local_size)

    def _launch(self, launch:str, args:Tuple[object, ...],
                kwargs:Optional[Dict[str, object]]=None)->object:
        """Shared body of the three launch styles, honouring the bracket selectors."""
        instance:object = self._instance
        if self._kernel._is_classmethod or self._kernel._self_param == "cls":
            # Same classmethod rule as __call__ above: the bound class travels when the
            # descriptor supplied one, an explicitly passed class is peeled, and
            # otherwise the owner resolves from the qualname.
            if instance is None and args and isinstance(args[0], type):
                instance = args[0]
                args = args[1:]
            call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args,
                                                        kwargs or {},
                                                        skip_first=True)
            return self._kernel._invoke(instance, call_args, self._backend, self._device,
                                    self._global_size, self._local_size, launch=launch)
        call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args,
                                                    kwargs or {},
                                                    skip_first=self._kernel._is_method)
        return self._kernel._invoke(
            instance, call_args, self._backend, self._device,
            self._global_size, self._local_size, launch=launch)

    @_native_argument_errors
    def submit(self, *args:object, **kwargs:object)->concurrent.futures.Future:
        """Non-blocking launch of the bracketed form, e.g. ``k["cuda"].submit(...)``."""
        return self._launch("submit", args, kwargs)

    @_native_argument_errors
    def async_call(self, *args:object, **kwargs:object)->asyncio.Future:
        """Non-blocking asyncio launch of the bracketed form, e.g. ``k["cuda"].async_call(...)``."""
        return self._launch("async", args, kwargs)


class _BoundKernelMethod(_KernelBase[_P]):
    """Bound instance-method form: instance.kernel_method(...)

    Reached through ``PyKernel.__get__``. Its parameters are deliberately loose: the
    decorated method's ``P`` still carries the receiver, which a type checker has no way to
    drop from a descriptor's return type, so spelling ``P`` here would make every correct
    ``obj.method(x, y)`` look like ``method(self=x, x=y)`` and report a missing argument.
    The receiver is also the one thing a caller never passes explicitly -- the class form
    ``Klass.method(obj, x, y)`` keeps its full signature, because that access goes through
    ``PyKernel`` itself.
    """

    def __init__(self, kernel:"PyKernel[_P]", instance:object):
        self._kernel:"PyKernel[_P]" = kernel
        self._instance:object = instance

    @property
    def __self__(self)->object:
        """The receiver this bound form carries.

        A python `MethodType` answers `__self__`, and the classmethod spellings in
        the tests (and any caller mirroring the protocol) read it back to get the
        receiver -- the class for a classmethod kernel, the instance for an
        instance method. Without it, `bound.__self__` raises AttributeError where
        the same access on a real bound method would have worked.
        """
        return self._instance

    @property
    def __func__(self)->"PyKernel[_P]":
        """The kernel the bound form wraps (a `MethodType` answers the same way).

        The classmethod-outermost spelling reaches the kernel's own surface through
        it: `bound.__func__._cache_dir(...)`, `bound.__func__.raw_func`, and the
        rest of the introspection a caller who holds only the bound form can ask
        for.
        """
        return self._kernel

    def __getattr__(self, name:str)->object:
        """Forward kernel-owning attributes (`_cache_dir`, `raw_func`, ...).

        The bound form is what callers hold, and the kernel's introspection
        surface (`_cache_dir` for the generated sources, `raw_func` for the
        decorated function, `_signature_stub` for the signature) lives on the
        kernel itself. `__getattr__` runs only when normal lookup fails, so the
        bound form's own attributes still win.
        """
        if name.startswith("__") or name in ("_kernel", "_instance"):
            raise AttributeError(name)
        return getattr(self._kernel, name)

    def __getitem__(self, item:object)->object:
        return _KernelCall(self._kernel, self._instance)[item]

    @_native_argument_errors
    # The result is the kernel body's value, produced on the device. `_R` is the
    # decorated function's result type but is not bound on this class, so the
    # honest annotation is the untyped one -- `object` would forbid unpacking a
    # tuple return at the call site.
    def __call__(self, *args:object, **kwargs:object)->Any:
        call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=True)
        if self._kernel._is_classmethod or self._kernel._self_param == "cls":
            # The receiver the descriptor bound is the class; passing it as the
            # leading argument is the supported spelling for a class the kernel's
            # qualname cannot resolve (a test-local class, say) -- the same shape
            # `_KernelCall`'s class form accepts.
            return self._kernel._invoke(self._instance, call_args, None, None,
                                        None, None)
        return self._kernel._invoke(self._instance, call_args, None, None, None, None)

    @_native_argument_errors
    def submit(self, *args:object, **kwargs:object)->concurrent.futures.Future:
        """Non-blocking launch: ``instance.kernel_method.submit(...)``."""
        call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=True)
        return self._kernel._invoke(self._instance, call_args, None, None, None, None,
                                    launch="submit")

    @_native_argument_errors
    def async_call(self, *args:object, **kwargs:object)->asyncio.Future:
        """Non-blocking asyncio launch: ``instance.kernel_method.async_call(...)``."""
        call_args:Tuple[object, ...] = _bind_arguments(self._kernel.raw_func, args, kwargs,
                                                    skip_first=True)
        return self._kernel._invoke(self._instance, call_args, None, None, None, None,
                                    launch="async")


# The name a host region's rewritten `return` raises, and the exception carrying its
# value (see _TailReturnRewriter and PyKernel._run_host_region).
_TAIL_RETURN_NAME:str = "_pygpgpu_tail_return"


class _TailReturn(BaseException):
    """Carries a host region's `return` out of the compiled statement that raised it.

    A region's statements are exec'd as a module body, where a `return` is a compile-time
    syntax error -- yet a `return` after the last magic loop is the ordinary way a body
    answers a value, and it may sit inside an `if` (`if n > 0: return out[0]`), which is
    exactly the shape that used to fail. Rewriting it as a raise keeps python's own
    control flow instead of approximating it: the value is evaluated where the `return`
    stands, an early return ends the enclosing loops, and a `finally` in the region still
    runs.

    It derives from `BaseException` rather than `Exception` on purpose: a region is user
    code, and an `except Exception` in it must not swallow the transfer that stands for
    its own `return`.
    """

    def __init__(self, value:object)->None:
        super().__init__()
        self.value:object = value


class _TailReturnRewriter(ast.NodeTransformer):
    """Rewrite every `return` of a host region into a `_TailReturn` raise.

    A nested function definition is left alone: its `return`s belong to *it*, not to the
    region, and rewriting them would turn a helper the region defines into one that
    raises. Nothing else in a region has its own scope.
    """

    def visit_Return(self, node:ast.Return)->ast.stmt:
        value:ast.expr = node.value if node.value is not None else ast.Constant(value=None)
        raised:ast.stmt = ast.Raise(
            exc=ast.Call(func=ast.Name(id=_TAIL_RETURN_NAME, ctx=ast.Load()),
                         args=[value], keywords=[]),
            cause=None)
        # The synthesized nodes carry no position of their own, and `compile` refuses an
        # expression without one: take the `return`'s own line so the region's traceback
        # still points at the statement the user wrote.
        ast.copy_location(raised, node)
        return ast.fix_missing_locations(raised)

    def visit_FunctionDef(self, node:ast.FunctionDef)->ast.AST:
        return node

    def visit_AsyncFunctionDef(self, node:ast.AsyncFunctionDef)->ast.AST:
        return node
