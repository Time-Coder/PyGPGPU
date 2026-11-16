"""Reading the `#include` directives of one source file, from its parse tree.

The preprocessor itself runs on pcpp, which works on raw text and emits its own
`#line` directives -- there is no tree to read there. But *which files a source
depends on* is a question about the source, and the grammars answer it exactly:

* a commented-out `#include` is not a `preproc_include` node, so a file nothing
  includes no longer enters the dependency set (a text pattern collected it);
* the `#  include` spelling with spaces needs no pattern to anticipate it;
* the quoted/angled distinction is a child node's type (`string_literal` versus
  `system_lib_string`), not a capture group.

A grammar that cannot be imported (a machine without this backend installed) falls
back to the textual scan, so this stays a refinement of the dependency set rather
than a new hard requirement.
"""
from __future__ import annotations

import re
from typing import List, Tuple


# The fallback pattern, used only when no grammar is importable. It answers the same
# `(name, quoted)` pairs the tree walk answers.
_INCLUDE_FALLBACK = re.compile(
    r'^\s*#\s*include\s*(?:"(?P<quoted>[^"]+)"|<(?P<angled>[^>]+)>)', re.MULTILINE)


def include_directives(content: str) -> List[Tuple[str, bool]]:
    """Every `#include` in one source, as `(path, is_quoted)` pairs.

    `is_quoted` says the directive was spelled `"..."`, which searches the including
    file's own directory first; an angled one searches only the include paths.
    """
    tree_reader = _grammar_reader()
    if tree_reader is not None:
        return tree_reader(content)
    return [(match.group("quoted") if match.group("quoted") is not None
             else match.group("angled"), match.group("quoted") is not None)
            for match in _INCLUDE_FALLBACK.finditer(content)]


def _grammar_reader():
    """A parser for the includes, or None when no grammar can be imported.

    The three C-family grammars all produce a `preproc_include` node holding either a
    `string_literal` or a `system_lib_string` child, so any one of them reads the
    directives of a file whose backend is not even installed. Metal is tried first
    only because `.metal` sources are the ones whose headers this package ships.
    """
    for module_name in ("tree_sitter_metal", "tree_sitter_opencl", "tree_sitter_cuda37",
                        "tree_sitter_cuda"):
        try:
            module = __import__(module_name)
            language = module.language()
        except Exception:
            continue
        try:
            from .ts_parser import make_parser

            parser = make_parser(language)
        except Exception:
            continue
        return _reader_for(parser)
    return None


def _reader_for(parser):
    def read(content: str) -> List[Tuple[str, bool]]:
        tree = parser.parse(content.encode("utf-8", "replace"))
        found: List[Tuple[str, bool]] = []
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "preproc_include":
                for child in node.named_children:
                    text: str = (child.text or b"").decode("utf-8", "replace").strip()
                    if child.type == "string_literal":
                        found.append((text.strip('"'), True))
                    elif child.type == "system_lib_string":
                        found.append((text.strip("<>"), False))
                continue
            stack.extend(node.named_children)
        # The walk is a stack, so the directives come back reversed; the order is what
        # makes the recursion visit a nested include before its sibling.
        found.reverse()
        return found
    return read
