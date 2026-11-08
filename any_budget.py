# -*- coding: utf-8 -*-
"""The `Any` budget: how many `Any` annotations the tree may carry, and where they are.

`Any` is a *permission to stop checking*, so it is the one annotation that needs a budget
rather than a rule nobody can enforce. This script lists every `Any` in an annotation, and
fails when the total crosses the recorded count -- so a new `Any` has to be paid for by
removing an old one, and a reviewer sees exactly which one was added.

**Two kinds of `Any` are counted separately, because they mean different things.**

- **A dtype slot.** `np.ndarray[Any, Any]` and `np.number[Any]` are numpy's own generic
  parameters: the slot accepts any dtype, and writing it says exactly that. It is not a
  refusal to type a value -- it is the type. These are counted and reported, because they
  are worth seeing, but they are not a budget item.
- **A hand-written `Any`.** `Dict[str, Any]`, `Callable[[str], Any]`, a bare `-> Any`: a
  shape this package could have named and did not. These are the ones the budget holds.

What counts as an annotation is a mention of `Any` in a function signature, a variable
annotation, or a type alias. A mention in a comment or a docstring does not count: the word
"any" in English is not an annotation.

Usage:
    python any_budget.py            # the report, and the check
    python any_budget.py --list     # every occurrence, with its file and line
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from typing import Dict, List, Tuple

#: The ceiling on the hand-written `Any` annotations. It is the count the tree carries
#: today; a new one has to be paid for by removing an old one.
BUDGET: int = 85

#: The kinds of file the budget covers.
SUFFIXES: Tuple[str, ...] = (".py",)

#: The folders the budget skips: the vendored third-party tree.
SKIP_DIRS: Tuple[str, ...] = ("pcpp",)

ROOT: str = "pygpgpu"

#: The numpy generic slots whose parameter is a dtype. An `Any` in one of them is numpy's
#: own generic parameter, not a hand-written refusal to type.
DTYPE_SLOTS = ("np.ndarray", "np.number", "np.generic", "np.integer", "np.floating",
               "np.complexfloating", "np.unsignedinteger", "np.signedinteger",
               "ndarray", "number", "generic")


def is_dtype_slot(node: ast.AST) -> bool:
    """Whether the `Any` sits in a numpy generic's dtype parameter.

    `np.ndarray[Any, Any]` is two dtype slots: the dtype and the shape. `np.number[Any]` is
    one. Either way the `Any` is numpy's own parameter.
    """
    if not isinstance(node, ast.Subscript):
        return False
    base = node.value
    name = ast.unparse(base) if base is not None else ""
    # `np.ndarray` written as an attribute chain, or as a string annotation
    short = name.split(".")[-1].strip("\"'")
    return short in DTYPE_SLOTS


def mentions_any(node: ast.AST) -> bool:
    """Whether the annotation mentions `Any`."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == "Any":
            return True
        if isinstance(child, ast.Attribute) and child.attr == "Any":
            return True
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if "Any" in child.value:
                return True
    return False


def count_hand_written(node: ast.AST) -> int:
    """How many hand-written `Any` the annotation carries.

    A `np.ndarray[Any, Any]` contributes none: both are dtype slots. A `Dict[str, Any]`
    contributes one.
    """
    if is_dtype_slot(node):
        return 0
    total = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == "Any":
            total += 1
        elif isinstance(child, ast.Attribute) and child.attr == "Any":
            total += 1
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            # a string annotation: parse it, so the dtype slots are told apart too
            try:
                inner = ast.parse(child.value, mode="eval").body
            except SyntaxError:
                if "Any" in child.value:
                    total += 1
                continue
            total += count_hand_written(inner)
    return total


def walk_file(path: str) -> List[Tuple[int, str, int]]:
    """The `Any` annotations in one file, as (line, text, hand-written count)."""
    with open(path, "r", encoding="utf-8") as handle:
        source: str = handle.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines: List[str] = source.split("\n")
    found: List[Tuple[int, str, int]] = []

    def record(node: ast.AST, line: int) -> None:
        text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        found.append((line, text, count_hand_written(node)))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for group in (args.posonlyargs, args.args, args.kwonlyargs):
                for argument in group:
                    if argument.annotation is not None and mentions_any(argument.annotation):
                        record(argument.annotation, argument.lineno)
            for extra in (args.vararg, args.kwarg):
                if extra is not None and extra.annotation is not None \
                        and mentions_any(extra.annotation):
                    record(extra.annotation, extra.lineno)
            if node.returns is not None and mentions_any(node.returns):
                record(node.returns, node.lineno)
        elif isinstance(node, ast.AnnAssign):
            if node.annotation is not None and mentions_any(node.annotation):
                record(node.annotation, node.lineno)
        elif isinstance(node, ast.Assign):
            # a type alias: `X = Union[..., Any]`
            for element in ast.walk(node.value):
                if isinstance(element, (ast.Tuple, ast.List)):
                    for inner in element.elts:
                        if mentions_any(inner):
                            record(inner, node.lineno)
                elif mentions_any(element):
                    record(element, node.lineno)
    return found


def collect() -> Dict[str, List[Tuple[int, str, int]]]:
    """Every `Any` annotation in the tree, keyed by its file."""
    found: Dict[str, List[Tuple[int, str, int]]] = {}
    for directory, subdirectories, names in os.walk(ROOT):
        subdirectories[:] = [d for d in subdirectories if d not in SKIP_DIRS]
        for name in sorted(names):
            if not name.endswith(SUFFIXES):
                continue
            path: str = os.path.join(directory, name)
            occurrences = walk_file(path)
            if occurrences:
                found[path] = occurrences
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="print every occurrence, not just the count")
    arguments = parser.parse_args()

    found = collect()
    hand_written = 0
    dtype_slots = 0
    for path in sorted(found):
        for line, text, count in found[path]:
            hand_written += count
            if count == 0:
                dtype_slots += 1

    if arguments.list:
        for path in sorted(found):
            for line, text, count in found[path]:
                kind = "dtype slot" if count == 0 else f"{count} hand-written"
                print(f"{path}:{line}: [{kind}] {text}")
        print(f"--- {hand_written} hand-written `Any`, {dtype_slots} numpy dtype slots")
        return 0

    print(f"any budget: ceiling {BUDGET} hand-written `Any` annotations, "
          f"the tree carries {hand_written} (plus {dtype_slots} numpy dtype slots)")
    for path in sorted(found):
        for line, text, count in found[path]:
            if count:
                print(f"  {path}:{line}: {text}")
    if hand_written > BUDGET:
        print(f"FAILED: {hand_written} hand-written `Any` annotations, the budget is "
              f"{BUDGET}. Name the real type, or remove an `Any` to make room.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
