# Traceback surgery: which frames are the user's and which are the library's.
from __future__ import annotations
import ast
import os
import sys
from types import TracebackType
from typing import (
    Callable,
    Dict,
    Optional,
    Tuple,
)
import numpy as np



def _is_user_frame(code:object)->bool:
    """True for a frame of code that lives outside this package."""
    filename:str = getattr(code, "co_filename", "") or ""
    if not filename:
        return False
    if filename.startswith("<") and filename.endswith(">"):
        # `-c`, stdin and exec'd code: no library of ours is written that way.
        return True
    package_dir:str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return not os.path.abspath(filename).startswith(package_dir + os.sep)


def _caller_traceback(traceback:Optional[TracebackType])->Optional[TracebackType]:
    """The first frame of ``traceback`` that is not inside this package."""
    if traceback is None:
        return None
    package_dir:str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prefix:str = package_dir + os.sep
    cursor:object = traceback
    while cursor is not None:
        filename:str = cursor.tb_frame.f_code.co_filename
        # `-c`, stdin and exec'd code use a pseudo-filename that abspath() would resolve
        # against the working directory, which can itself sit under the package.
        pseudo:bool = filename.startswith("<") and filename.endswith(">")
        if pseudo or not os.path.abspath(filename).startswith(prefix):
            return cursor
        cursor = cursor.tb_next
    return None


def _user_frame_syntax_error(error:SyntaxError)->SyntaxError:
    """Drop the library frames from a SyntaxError raised while translating a kernel.

    Kept for callers that want a trimmed chain rather than an empty one; the launch path
    itself clears the traceback (see _translate_errors), because a re-raise always prepends
    the frame doing the re-raising.
    """
    traceback = error.__traceback__
    if traceback is None:
        return error

    package_root:str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_prefix:str = package_root + os.sep

    keep:object = None
    cursor:object = traceback
    while cursor is not None:
        filename:str = cursor.tb_frame.f_code.co_filename
        # `-c`, stdin and exec'd code report a pseudo-filename that abspath() would resolve
        # against the working directory -- which can sit under the package; those frames are
        # the caller's, not the library's.
        pseudo:bool = filename.startswith("<") and filename.endswith(">")
        if pseudo or not os.path.abspath(filename).startswith(package_prefix):
            keep = cursor
            break
        cursor = cursor.tb_next
    if keep is None:
        return error
    error.__traceback__ = keep
    return error


# A comment constant that marks the frame stand-in built by _body_frame, so the entry
# wrapper can recognize a body failure among the errors it forwards.
_FRAME_MARKER = "__pygpgpu_body_frame__"


def _is_body_stand_in(code:object)->bool:
    """True for the code object :func:`_body_frame` compiles.

    The stand-in is the only function this library creates inside the user's own file, so
    its presence in a traceback identifies a body failure -- and it is the frame the
    caller's own python would have shown. The marker is a constant in the compiled body,
    which no ordinary function carries.
    """
    for constant in getattr(code, "co_consts", ()) or ():
        if isinstance(constant, str) and constant == _FRAME_MARKER:
            return True
    return False


def _subscript_stand_in(error:BaseException, filename:str, lineno:int, column:int,
                        func_name:str)->Optional[Tuple[str, Dict[str, object]]]:
    """A stand-in that really performs the too-wide subscript.

    CPython draws a caret from the failing *instruction*'s position, and the only way to get
    the one python would draw -- the object under `~`, the index under `^` -- is for the
    failing instruction to be the same subscript. So this builds a function whose last
    statement is the user's own expression, with the array bound to one of the real rank so
    numpy raises the identical ``IndexError``.

    Returns ``(source, namespace)``, or None when the recorded expression cannot be
    reproduced -- a target that is not a plain name (``self.items[i]``), or a slice whose
    own names cannot be bound -- in which case the caller falls back to a plain raise.
    """
    target:str = str(getattr(error, "subscript_text", "") or "")
    rank:int = int(getattr(error, "array_rank", 0) or 0)
    if not target or rank <= 0 or "[" not in target or not target.endswith("]"):
        return None
    base:str = target.split("[", 1)[0]
    if not base.isidentifier():
        return None
    try:
        parsed:ast.AST = ast.parse(target, mode="eval")
    except SyntaxError:
        return None
    # Every name the expression mentions stands for an axis; the array itself gets a shape
    # of the real rank, which is what makes numpy refuse the subscript with python's own
    # words and mark the position CPython would.
    namespace:Dict[str, object] = {node.id: 0 for node in ast.walk(parsed)
                                  if isinstance(node, ast.Name)}
    namespace[base] = np.zeros((1,) * rank, dtype=np.float32)
    padding:str = " " * max(column, 0)
    source:str = ("\n" * max(lineno - 3, 0)
                  + f"def {func_name}():\n"
                  + padding + f"{_FRAME_MARKER!r}\n"
                  + padding + target + " = 1.0\n")
    return source, namespace


def _body_frame(func:Callable, error:BaseException)->Optional[TracebackType]:
    """A frame standing in for ``func``'s own body at the line that failed.

    Native python would show exactly two: the caller's line and the failing line inside the
    function. The body's real frame cannot simply be kept, because it is a frame of the
    interpreter backend's machinery rather than of the user's code -- and the device
    backends never execute the body at all.

    A traceback frame's ``co_filename``/``tb_lineno`` are read-only, so the stand-in is a
    function compiled from source carrying the user's filename and enough blank lines to
    place its ``raise`` on the failing line. Its code positions then supply the same caret
    CPython would draw. Only the mapping matters: the file, line and column are what a
    reader sees.
    """
    filename:str = getattr(getattr(func, "__code__", None), "co_filename", "") or ""
    if not filename or filename.startswith("<"):
        return None
    # The line the error points at, in the user's file. For an unpacking failure that is the
    # `for` statement itself, which is exactly what python blames. A *device* backend refuses
    # that shape before the body ever runs, so there is no frame to read it from -- the
    # translator records the position on the exception instead (see _unpacking_error).
    qualname_for_frame:str = (getattr(func, "__qualname__", None)
                              or getattr(func, "__name__", "kernel"))
    body_name:str = qualname_for_frame.rsplit(".", 1)[-1]

    lineno:int = int(getattr(error, "lineno", 0) or 0)
    if not lineno:
        cursor:object = error.__traceback__
        while cursor is not None:
            frame_code:object = cursor.tb_frame.f_code
            if (os.path.abspath(frame_code.co_filename) == os.path.abspath(filename)
                    and frame_code.co_name == body_name
                    and not _is_body_stand_in(frame_code)):
                # The interpreter backend ran the body natively, so the frames BELOW this
                # one are the user's own too -- a raise inside a helper shows the helper's
                # frame, exactly as an undecorated call would. Keep the chain from this
                # frame down to the last user frame: a library frame after it (a numpy
                # wrapper, a callback) is where a native traceback would stop as well.
                # Nothing is gained by replacing these frames with a stand-in, and the
                # caret is lost by doing so, which is what made this backend's error read
                # differently from the undecorated one.
                inner:object = cursor
                while inner.tb_next is not None \
                        and _is_user_frame(inner.tb_next.tb_frame.f_code):
                    inner = inner.tb_next
                inner.tb_next = None
                return cursor
            cursor = cursor.tb_next
        cursor = error.__traceback__
        while cursor is not None:
            if os.path.abspath(cursor.tb_frame.f_code.co_filename) == os.path.abspath(filename):
                lineno = cursor.tb_lineno
                break
            cursor = cursor.tb_next
    if not lineno:
        return None

    qualname:str = getattr(func, "__qualname__", None) or getattr(func, "__name__", "kernel")
    # A nested function's qualname carries its outer names; the frame shows the last one.
    func_name:str = qualname.rsplit(".", 1)[-1]
    # The caret's start column: what python underlines is the offending expression, and the
    # translator records where it begins (see `_check_shape_indices`). A line-only message
    # would point at the statement with no indication of *which* part of it failed.
    column:int = max(int(getattr(error, "column", 0) or 0), 0)
    # The statements need the block's own four spaces *plus* the caret's column, so an
    # error with no recorded column still compiles as an ordinary indented body.
    padding:str = " " * (column + 4)
    # The marker is a real string CONSTANT in the code object, which is what identifies this
    # stand-in later (see _is_body_stand_in). It sits on the line *above* the raise so the
    # raise stays on `lineno`; a comment would not reach `co_consts`.
    source:str = ("\n" * max(lineno - 3, 0)
                  + f"def {func_name}():\n"
                  + padding + f"'{_FRAME_MARKER}'\n"
                  + padding + "raise _PYGPGPU_FRAME_EXCEPTION\n")
    # Prefer a stand-in that performs the real subscript, because that is the only way to
    # get CPython's own caret; the plain raise below is the fallback.
    reproduced:Optional[Tuple[str, Dict[str, object]]] = _subscript_stand_in(
        error, filename, lineno, column, func_name)
    if reproduced is not None:
        source, namespace = reproduced
    else:
        namespace = {"_PYGPGPU_FRAME_EXCEPTION": error}
    try:
        exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - a frame stand-in
        frame_func:object = namespace[func_name]
    except (SyntaxError, ValueError):
        return None
    try:
        frame_func()
    except BaseException:  # noqa: BLE001 - the carrier raises; the real error travels along
        # Only the frame for the user's file is wanted: the caller's frame comes from
        # `_caller_traceback` and everything else in this chain is this module's own.
        cursor = sys.exc_info()[2]
        while cursor is not None:
            if os.path.abspath(cursor.tb_frame.f_code.co_filename) == os.path.abspath(filename):
                # The captured chain continues into this function and its caller, which are
                # library frames -- the stand-in is a whole traceback, not just its head.
                cursor.tb_next = None
                return cursor
            cursor = cursor.tb_next
    return None


def _user_traceback(error:BaseException, func:Callable)->Optional[TracebackType]:
    """The traceback native python would have produced: the caller, then the failing line.

    Everything between the two belongs to this library, so it is spliced out. The head is
    the caller's own frame -- the line that asked the kernel to run, added by the
    re-raise at the launch boundary -- and its successor is the chain
    :func:`_body_frame` returns: the stand-in it built for the body's failing line on a
    device backend, and the body's own real frames down to the raise on the interpreter
    backend (a raise inside a helper keeps the helper's frame, exactly as an undecorated
    call would show it).

    The chain ``_body_frame`` returns is already cut at the last user frame, so it is
    handed back unchanged: a traceback node cannot be fabricated, and the nodes it carries
    are either the stand-in's own raise or the live chain of the error being handled --
    both real raises, both already cut. When there is no body frame at all (the body never
    ran, or its source has no file to name), the first non-library frame of the live chain
    is what remains.
    """
    body:Optional[TracebackType] = _body_frame(func, error)
    if body is None:
        return _caller_traceback(error.__traceback__)
    return body
