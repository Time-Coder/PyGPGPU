# The errors a translation raises, positioned at the user's own source line.
from __future__ import annotations
import ast
from typing import (Optional, Tuple)



def _syntax_error(message:str, node:Optional[ast.AST]=None,
                  filename:str="<kernel>",
                  source:Optional[str]=None,
                  first_line:int=1)->SyntaxError:
    """Build a plain :class:`SyntaxError` positioned at kernel source the backend rejects.

    Callers ``raise`` the result ``from None``, so the traceback shows only the user's own
    frame and CPython's familiar rendering: the file, the line number, the offending source
    line and a caret under the column.

    The translator parses a *dedented copy* of the function with its decorator line removed,
    so an AST ``lineno`` counts from that copy. ``first_line`` converts it back to the line
    in the user's file; the copy's line 1 is the decorator line.
    """
    lineno:int = getattr(node, "lineno", 0) if node is not None else 0
    column:int = getattr(node, "col_offset", 0) if node is not None else 0
    end_column:Optional[int] = getattr(node, "end_col_offset", None) if node is not None else None

    # lineno in the parsed copy -> line in the user's file (the copy starts at the decorator).
    file_lineno:Optional[int] = (first_line + lineno - 1) if lineno else None

    text:Optional[str] = None
    # Prefer the user's own file: the parsed copy is dedented and has no decorator, so its
    # line text would show the wrong indentation and the caret would sit off by that much.
    if filename and filename not in ("<kernel>", "<string>"):
        try:
            with open(filename, encoding="utf-8") as handle:
                file_lines = handle.read().splitlines()
        except OSError:
            file_lines = []
    else:
        file_lines = []

    if file_lines:
        if file_lineno is not None and 0 < file_lineno <= len(file_lines):
            text = file_lines[file_lineno - 1]
            # Columns come from the dedented copy; add back the indentation it removed. The
            # same shift applies to the node's end column.
            if source and lineno and 0 < lineno <= len(source.splitlines()):
                shift = max(len(text) - len(source.splitlines()[lineno - 1]), 0)
                column += shift
                if end_column is not None:
                    end_column += shift
    elif source and lineno:
        lines = source.splitlines()
        if 0 < lineno <= len(lines):
            text = lines[lineno - 1]

    details:Tuple[object, ...] = (
        filename if filename else "<kernel>",
        file_lineno,
        (column + 1) if text is not None else None,
        text,
    )
    error = SyntaxError(message, details)
    if end_column is not None and text is not None:
        # A column range makes CPython underline the whole span instead of one caret.
        error.end_lineno = file_lineno
        error.end_offset = end_column + 1
    return error


def _unpack_count_message(names:int, values:int)->str:
    """What python itself says when a loop's names do not match the number of values.

    `for i, j in np.ndindex(arr.shape[0])` is an ordinary unpacking error in python -- the
    loop asks for more names than the iterator yields -- and a kernel body should say the
    same thing, in the same words, so the two spellings of one mistake read alike. CPython
    has two messages and picks by direction: too few values is "not enough values to
    unpack", too many is "too many values to unpack", both with "(expected N, got M)".
    """
    if values < names:
        return (f"not enough values to unpack (expected {names}, got {values})")
    return (f"too many values to unpack (expected {names}, got {values})")


def _unpacking_error(message:str, node:Optional[ast.AST], filename:str,
                     source:Optional[str]=None, first_line:int=1)->ValueError:
    """An unpacking :class:`ValueError`, positioned at the loop that raises it.

    ``ValueError`` has no ``filename``/``lineno`` of its own, so the position travels in the
    message exactly as the interpreter's does -- and the traceback is trimmed to the user's
    frame by the launch (see ``PyKernel._invoke``). The source line and caret come from the
    same helper a translation error uses, so the two render alike.
    """
    error:ValueError = ValueError(message)
    _position_error(error, node, filename, source, first_line)
    return error


def _position_error(error:BaseException, node:Optional[ast.AST], filename:str,
                    source:Optional[str]=None, first_line:int=1)->BaseException:
    """Attach file/line/column to an exception raised *about* a span of the user's source."""
    if node is None:
        return error
    lineno:int = getattr(node, "lineno", 0)
    file_lineno:Optional[int] = (first_line + lineno - 1) if lineno else None
    error.filename = filename  # type: ignore[attr-defined]
    error.lineno = file_lineno  # type: ignore[attr-defined]
    return error
