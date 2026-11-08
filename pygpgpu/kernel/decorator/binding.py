# Binding a call's arguments against the kernel's signature.
from __future__ import annotations
from ._typing import _P, _R
from .frames import _is_body_stand_in, _is_user_frame
import functools
import inspect
from types import TracebackType
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)



def _bind_arguments(func:Callable, args:Tuple[object, ...],
                    kwargs:Dict[str, object],
                    skip_first:bool=False)->Tuple[object, ...]:
    """Turn a call into the positional tuple the launch path expects.

    A kernel is launched positionally: the argument order decides the device parameter
    order, and the work size is read from a named argument. Accepting keywords here
    keeps the call site looking like ordinary Python without pushing that bookkeeping
    into the launch path. Omitted parameters with a default are filled in, so the launch
    always sees one value per device parameter.

    ``func`` is the undecorated function, so the signature is the author's own.
    ``skip_first`` drops the leading parameter of a method, whose receiver is supplied
    separately by the caller.

    A bad call must fail the way the same call to the undecorated function fails, so the
    binding runs through CPython itself (see _signature_stub) instead of a re-worded
    message of our own: the wording, the singular/plural forms, the "Did you mean ...?"
    suggestion and even the mention of keyword-only leftovers then follow the running
    interpreter rather than a hand-written table of phrases.

    A method is checked with its receiver counted, because that is how CPython words the
    error for the undecorated method: `Model.run() takes 3 positional arguments but 4 were
    given`. The receiver is a placeholder -- its value never reaches the kernel, only the
    count does.
    """
    full_signature:Optional[inspect.Signature] = _callable_signature(func, skip_first=False)
    signature:Optional[inspect.Signature] = _callable_signature(func, skip_first)
    if signature is None:
        if kwargs:
            raise TypeError(
                "keyword arguments need a retrievable signature; call positionally") from None
        return args

    # Let CPython bind and reject: it raises the native TypeError for an arity or keyword
    # mismatch, attributed to the kernel's own qualified name and trimmed to the caller.
    assert full_signature is not None
    stub:Callable = _signature_stub(full_signature, func)
    bound:inspect.BoundArguments = _bind_and_reject(
        func, signature, stub, args, kwargs, skip_first)

    ordered:List[object] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in bound.arguments:
            ordered.append(bound.arguments[name])
        else:
            # bind() accepted the call without it, so the parameter has a default.
            ordered.append(parameter.default)
    return tuple(ordered)


def _bound_arguments_for_call(func:Callable, args:Tuple[object, ...],
                              kwargs:Dict[str, object],
                              skip_first:bool=False)->Tuple[object, ...]:
    """``_bind_arguments`` as the entry points call it (see _native_argument_errors)."""
    return _bind_arguments(func, args, kwargs, skip_first)


def _native_argument_errors(entry:Callable[_P, _R])->Callable[_P, _R]:
    """Give an entry point native-python's single-frame traceback for a bad call.

    Native python blames the call site and nothing else; a decorated call would otherwise
    also show this module's ``__call__`` and the signature stub, so the same mistake looked
    different once a function was decorated. The trimming happens here, at the entry point,
    because clearing the traceback deeper inside and re-raising lets every frame the
    exception travels through push itself back on again.

    The parameters are forwarded unchanged (``_P``/``_R``), so wrapping an entry point does
    not hide its signature from a type checker: ``functools.wraps`` alone conservatively
    reports ``(*args, **kwargs)``, which would undo the parameter hints the caller's own
    ``def`` gave the kernel.

    A failure of the *body* travels through here too, and arrives already carrying a
    stand-in for the body's own line (see ``PyKernel._invoke``). The wrapper is the last
    place its own frame can be dropped, because a bare ``raise`` here would push it back on:
    only the stand-in is kept, so the reader sees the caller's line and the body's line and
    nothing of this library.
    """
    @functools.wraps(entry)
    def wrapper(*args:_P.args, **kwargs:_P.kwargs)->_R:
        try:
            return entry(*args, **kwargs)
        except TypeError as error:
            error.__traceback__ = None
            raise
        except Exception as error:
            body:Optional[TracebackType] = _body_only(error)
            if body is None:
                raise
            error.__traceback__ = body
            raise

    return wrapper


def _body_only(error:BaseException)->Optional[TracebackType]:
    """The exceptions raised by a kernel body, without this library's frames.

    ``PyKernel._invoke`` ends the chain with the body's own frame: a stand-in on a device
    backend, and the *real* frame on the interpreter backend, which has column information
    a stand-in cannot carry. Everything ahead of it belongs to the launch machinery.
    Nothing is returned for an error that carries neither -- a failure raised before the
    body ran -- so that one keeps its own diagnostic.
    """
    traceback:object = error.__traceback__
    keep:object = None
    cursor:object = traceback
    while cursor is not None:
        code:object = cursor.tb_frame.f_code
        if _is_body_stand_in(code) or _is_user_frame(code):
            # The last such frame is the body's: the launch machinery is all inside this
            # package, so anything after it would be a callee, and the body is the callee.
            keep = cursor
        cursor = cursor.tb_next
    return keep


def _callable_signature(func:Callable, skip_first:bool)->Optional[inspect.Signature]:
    """The signature a kernel call is checked against, with a method's receiver removed."""
    try:
        signature:inspect.Signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    if not skip_first:
        return signature
    parameters:List[inspect.Parameter] = list(signature.parameters.values())[1:]
    return signature.replace(parameters=parameters)


_STUB_SCACHE:Dict[str, Callable] = {}

# Stands in for a method's receiver when CPython is asked to word an argument error.
# Never reaches the kernel: it exists so the error counts the receiver like native python.
_RECEIVER_PLACEHOLDER:object = object()


def _signature_stub(signature:inspect.Signature, func:Callable)->Callable:
    """A stand-in function with this signature, used only to raise CPython's own errors.

    Calling it performs the real argument binding, so a wrong call fails with exactly the
    text the undecorated function would have produced -- including the parts a
    reimplementation tends to miss ("but 3 were given", "1 required positional argument",
    "Did you mean 'b'?"). Annotations are dropped because they cannot always be
    stringified, and no parameter type reaches this function: only the shape matters.

    Stubs are cached per signature text and qualified name, so a repeated bad call does
    not recompile the same throwaway function.
    """
    source_signature:str = str(signature)
    qualname:str = _qualname_for_errors(func)
    cache_key:str = qualname + source_signature
    stub:Optional[Callable] = _STUB_SCACHE.get(cache_key)
    if stub is not None:
        return stub

    body:str = ("from __future__ import annotations\n"
                f"def _signature_stub{source_signature}:\n"
                f"    return None\n")
    namespace:Dict[str, object] = {"__name__": "pygpgpu.kernel._signature_stub"}
    exec(compile(body, "<kernel signature>", "exec"), namespace)  # noqa: S102 - a builder, not a container
    stub = namespace["_signature_stub"]
    # CPython prints the qualified name in its message, so an error reads
    # `Model.method() ...` exactly as it would for the undecorated method.
    stub.__qualname__ = qualname
    stub.__name__ = qualname
    _STUB_SCACHE[cache_key] = stub
    return stub


def _qualname_for_errors(func:Callable)->str:
    """The name CPython would print for this callable in an argument error."""
    qualname:str = getattr(func, "__qualname__", None) or getattr(func, "__name__", "kernel")
    return qualname


def _bind_and_reject(func:Callable, signature:inspect.Signature,
                     stub:Callable, args:Tuple[object, ...],
                     kwargs:Dict[str, object], skip_first:bool)->inspect.BoundArguments:
    """Let CPython bind the call, re-raising its TypeError from the caller's own frame.

    Native python reports a bad call with one frame -- the call site -- and the traceback
    here would otherwise also show this module's ``__call__`` and the stub's call, which is
    what makes the same mistake look different once a function is decorated. Clearing the
    traceback and re-raising at this boundary leaves exactly the caller's frame, the same
    shape ``_translate_errors`` gives a translation failure.
    """
    try:
        if skip_first:
            stub(_RECEIVER_PLACEHOLDER, *args, **kwargs)
        else:
            stub(*args, **kwargs)
        return signature.bind(*args, **kwargs)
    except TypeError as error:
        error.__traceback__ = None
        raise
