# -*- coding: utf-8 -*-
"""The duplicate-binding check: one module-level name is bound once.

A definition that lands twice in one module is a copy-paste artifact -- a block left behind
when a module was split, or a header pasted a second time. Nothing objects at runtime (the
last assignment wins, and the two usually hold the same value), and no linter sees it:
ruff's F811 covers a redefined import or `def`, never a module-level assignment. So it
survives review, and what it leaves behind reads as a comment attached to the wrong name --
the dead copy takes the comment that belongs to the live one.

The check is deliberately narrow, because "assigned twice" is not by itself a defect: a
module may legitimately rebind a name. It flags the one case that cannot be intended -- the
same name bound twice, unconditionally, to the *same* value. A name re-bound to a different
value is left alone (a generator script does that), and a binding inside `if`/`try` is
skipped entirely, because that is how a `TYPE_CHECKING` pair spells one name twice on
purpose.

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


def main() -> int:
    root: str = os.path.dirname(os.path.abspath(__file__))
    violations: List[str] = []
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

    print(f"duplicate bindings: {files} files scanned")
    if violations:
        print("\nbound twice to the same value, in one module's own body:")
        for entry in violations:
            print("  " + entry)
        print("\nFAILED: delete the dead copy, and move its comment onto the live one.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
