# -*- coding: utf-8 -*-
"""The duplicate-binding check: a declaration is written once.

A definition that lands twice is a copy-paste artifact -- a block left behind when a module
was split, or a header pasted a second time. Nothing objects at runtime (the last assignment
wins, and the two usually hold the same value), and no linter sees it: ruff's F811 covers a
redefined import or `def`, never a module-level assignment or a dict entry. So it survives
review, and what it leaves behind reads as a comment attached to the wrong name -- the dead
copy takes the comment that belongs to the live one.

Two shapes are checked, because they are the same failure and both are silent. A
**module-level name** may be bound once: the check is deliberately narrow, because "assigned
twice" is not by itself a defect -- a module may legitimately rebind a name -- so it flags
the one case that cannot be intended, the same name bound twice, unconditionally, to the
*same* value. A name re-bound to a different value is left alone (a generator script does
that), and a binding inside `if`/`try` is skipped entirely, because that is how a
`TYPE_CHECKING` pair spells one name twice on purpose.

A **dict literal** may bind one key once, and a repeated key is never intended: python keeps
the last value, so the earlier entry vanishes without a word. That is how the OpenCL error
tables lost one of two hints for a code (`ErrorCode.CL_INVALID_VALUE` listed twice in one
literal), and how a query table lost a row. Only a key the source spells -- a literal, a
name, an attribute chain such as `ErrorCode.CL_INVALID_VALUE` -- is compared; a computed key
is left alone, because `d[f(i)] = x` twice is a loop written out, not a duplicate.

Usage:
    python dup_defs.py    the check (exit 1 on a violation)
"""

from __future__ import annotations

import ast
import os
import sys
from typing import Dict, Iterator, List, Tuple

# Directories that are never walked: version control, tool caches, build output, the
# vendored C preprocessor, and the generated compile caches.
SKIP_DIRECTORIES: Tuple[str, ...] = (
    ".git", ".ruff_cache", ".pytest_cache", "__pycache__", "pygpgpu.egg-info",
    "build", "dist", "pcpp", "__clcache__", "__cucache__", "__hipcache__", "__metalcache__",
)

# Path fragments (posix-style, so this works on either platform) that opt a file out.
EXEMPT_FRAGMENTS: Tuple[str, ...] = (
    "/pcpp/",   # the vendored preprocessor
)

# One module-level binding: the name, the line it sits on, its value expression, and the
# annotation's structural dump ("" when it carries none).
Binding = Tuple[str, int, str, str]

# One dict key bound twice in one literal: how the key is spelled, the first line, the repeat.
RepeatedKey = Tuple[str, int, int]


def source_files(root: str) -> Iterator[Tuple[str, str]]:
    """Yield (relative posix path, absolute path) for every python file the check applies to."""
    for directory, sub_directories, names in os.walk(root):
        sub_directories[:] = [name for name in sub_directories if name not in SKIP_DIRECTORIES]
        for name in names:
            if not name.endswith(".py"):
                continue
            full: str = os.path.join(directory, name)
            relative: str = os.path.relpath(full, root).replace(os.sep, "/")
            if any(fragment in "/" + relative for fragment in EXEMPT_FRAGMENTS):
                continue
            yield relative, full


def module_bindings(tree: ast.Module) -> List[Binding]:
    """Every binding in the module's own body.

    A statement under `if`, `try`, `class` or `def` is not a module-level binding: the first
    two exist precisely to choose one of two bindings at import time, and the last two open a
    scope of their own. Only `Assign` and `AnnAssign` bind a name at this level.
    """
    found: List[Binding] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets: List[ast.expr] = list(statement.targets)
            annotation: ast.expr | None = None
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            annotation = statement.annotation
        else:
            continue
        if statement.value is None:
            continue
        value: str = ast.dump(statement.value)
        annotation_dump: str = ast.dump(annotation) if annotation is not None else ""
        for target in targets:
            if isinstance(target, ast.Name):
                found.append((target.id, statement.lineno, value, annotation_dump))
    return found


def duplicate_bindings(bindings: List[Binding]) -> List[Tuple[str, List[Binding]]]:
    """The names bound more than once to one value, with every binding that names them."""
    by_name: Dict[str, List[Binding]] = {}
    for binding in bindings:
        by_name.setdefault(binding[0], []).append(binding)
    duplicated: List[Tuple[str, List[Binding]]] = []
    for name, places in by_name.items():
        if len(places) < 2:
            continue
        if len({place[2] for place in places}) == 1:
            duplicated.append((name, places))
    return duplicated


def key_text(key: ast.expr) -> str | None:
    """How a dict key is spelled in the message; None when it is a computed expression."""
    if isinstance(key, ast.Constant):
        return repr(key.value)
    if isinstance(key, ast.Name):
        return key.id
    if isinstance(key, ast.Attribute):
        owner: str | None = key_text(key.value)
        return key.attr if owner is None else owner + "." + key.attr
    return None


def repeated_keys(tree: ast.Module) -> List[RepeatedKey]:
    """Every dict literal that binds one key expression twice, anywhere in the file.

    A repeated key is never intended -- the later value silently replaces the earlier one --
    so this needs no "same value" qualification the way a rebinding does. `**mapping`
    entries carry no key at all and are skipped, and a computed key is left alone: a
    function call or a subscript is how a loop filling a dict is written out.
    """
    found: List[RepeatedKey] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        first: Dict[str, int] = {}
        for key in node.keys:
            if key is None:
                continue
            text: str | None = key_text(key)
            if text is None:
                continue
            if text in first:
                found.append((text, first[text], key.lineno))
            else:
                first[text] = key.lineno
    found.sort(key=lambda entry: entry[2])
    return found


def main() -> int:
    root: str = os.path.dirname(os.path.abspath(__file__))
    violations: List[str] = []
    keys: List[str] = []
    files: int = 0

    for relative, full in source_files(root):
        files += 1
        with open(full, "rb") as handle:
            # `utf-8-sig`, because several files in the tree start with a BOM and
            # `ast.parse` refuses one on a str (python itself strips it when it reads).
            source: str = handle.read().decode("utf-8-sig")
        try:
            tree: ast.Module = ast.parse(source, filename=relative)
        except SyntaxError as error:
            violations.append(f"{relative}: cannot be parsed ({error.msg})")
            continue
        for name, places in duplicate_bindings(module_bindings(tree)):
            lines: str = ", ".join(str(place[1]) for place in places)
            annotations: set = {place[3] for place in places}
            note: str = " (annotations differ)" if len(annotations) > 1 else ""
            violations.append(f"{relative}: '{name}' bound at lines {lines}{note}")
        for text, first_line, repeat in repeated_keys(tree):
            keys.append(f"{relative}: '{text}' bound at lines {first_line}, {repeat}")

    print(f"duplicate definitions: {files} files scanned")
    if not violations and not keys:
        print("OK")
        return 0
    if violations:
        print("\nbound twice to the same value, in one module's own body:")
        for entry in violations:
            print("  " + entry)
    if keys:
        print("\none key bound twice in one dict literal, so the earlier value is lost:")
        for entry in keys:
            print("  " + entry)
    print("\nFAILED: delete the dead copy and move its comment onto the live one; in a dict,")
    print("merge the two entries into one, or the earlier value disappears without a word.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
