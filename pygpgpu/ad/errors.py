# The refusals the AD stage raises, positioned like the translator's own.
from __future__ import annotations

import ast
from typing import Optional

from ..kernel.translator.errors import _syntax_error


def ad_error(message:str, node:Optional[ast.AST], filename:str="<kernel>",
             source:Optional[str]=None, first_line:int=1)->SyntaxError:
    """A :class:`SyntaxError` positioned at the user's own source line.

    The transform preserves the original nodes' line numbers, so an error about a
    statement the user wrote lands on that statement, with the caret under it -- the same
    rendering the translator's refusals get (see ``translator.errors._syntax_error``).
    """
    return _syntax_error(message, node, filename, source, first_line)


def ad_refusal(message:str, node:Optional[ast.AST])->SyntaxError:
    """A refusal whose position the caller has already resolved into the message."""
    error:SyntaxError = SyntaxError(message)
    if node is not None and getattr(node, "lineno", None):
        error.lineno = int(node.lineno)
    return error
