# Turning a `with inline, kernel(...)` block into a function the translator can read.
"""A block is a `@kernel` body with no `def`, so this module writes the `def` for it.

The generated text keeps the user's **absolute** line numbers: the `def` sits on the line of
the `with` statement, so every body statement stays at its own file line and column. The
translator maps an AST line back to the file by adding a constant, and this makes that
constant one -- a translation error therefore points at the block's real line, with the
real source text under the caret.

The block's data is not written into the call. The names the body reads from the enclosing
function become **parameters** of the rebuilt function -- the same thing a `@kernel` body
gets from `def add(a, b, out)` -- so everything downstream (argument binding, the copy
policy, outer globals, host regions, caching) is the machinery that already exists.
"""
from __future__ import annotations

import ast
import types
from typing import Dict, List

from ..translator.source_text import register_block_source


def block_name(with_line:int)->str:
    """The generated function's name. A `with` line is unique in a file, so this is too."""
    return f"inline_block_{with_line}"


def _with_indent(source_lines:List[str], node:ast.With)->int:
    """The indentation of the line the `with` statement starts on."""
    line:str = source_lines[node.lineno - 1]
    return len(line) - len(line.lstrip())


def block_body_lines(source_lines:List[str], node:ast.With)->List[str]:
    """The block's own lines, dedented by the `with` statement's indentation.

    A line with no indentation of its own (a blank line, or the second line of a
    triple-quoted string starting at column 0) is kept as it is: dedenting it would change
    the string, and a blank line is legal at any indentation.
    """
    indent:int = _with_indent(source_lines, node)
    prefix:str = " " * indent
    body:List[str] = []
    for line in source_lines[node.body[0].lineno - 1:node.end_lineno]:
        if line.startswith(prefix):
            body.append(line[indent:])
        else:
            body.append(line.lstrip() if line.strip() else "")
    return body


def block_text(source_lines:List[str], node:ast.With, parameters:List[str])->str:
    """The generated module: one `def` on the `with` line holding the block verbatim.

    Everything before the `def` is blank, so the generated text's line N is the user's file
    line N. That equality is the whole point of building the text this way.
    """
    name:str = block_name(node.lineno)
    lines:List[str] = [""] * (node.lineno - 1)
    lines.append(f"def {name}({', '.join(parameters)}):")
    lines.extend(block_body_lines(source_lines, node))
    return "\n".join(lines) + "\n"


def compile_only(text:str, filename:str)->types.CodeType:
    """Compile generated text without running it, for the name analysis pass."""
    return compile(text, filename, "exec")


def build_function(text:str, filename:str, with_line:int,
                   globals_dict:Dict[str, object])->types.FunctionType:
    """Compile the generated text and return the block's function.

    The function is rebuilt with the module's **real** globals dictionary, not a copy, so a
    module global resolves, a device helper defined in the module resolves, and a
    `global x` assignment in the block lands where it would have without the `with`. The
    code object carries the user's filename and the block's own line numbers, so what comes
    back is the function a `@kernel` definition would have produced, apart from its text.
    """
    name:str = block_name(with_line)
    namespace:Dict[str, object] = {"__name__": "__inline_block__",
                                   "__builtins__": __builtins__}
    exec(compile(text, filename, "exec"), namespace)  # noqa: S102 - the user's own block
    function:types.FunctionType = namespace[name]  # type: ignore[assignment]
    func:types.FunctionType = types.FunctionType(function.__code__, globals_dict, name)
    func.__qualname__ = name
    func.__module__ = str(globals_dict.get("__name__", "__inline_block__"))
    # The text holds blank lines before the `def`, so parsed-copy line 1 IS file line 1 --
    # which is exactly what `_source_first_line` has to answer for a block.
    register_block_source(func, text)
    return func
