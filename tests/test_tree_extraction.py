"""What the tree answers that a pattern over the text did not.

Three extraction sites moved from a regular expression to the parse tree or to the
python AST. Each test below pins the *difference* the change makes, not just the
happy path -- a pattern that agrees with the tree on ordinary input would pass a
test that only checked ordinary input.

The three:

* `Metal` reads its declarations from `tree-sitter-metal` instead of a scanner;
* the `#include` dependency scan reads `preproc_include` nodes;
* `range()`'s step is read from the python node instead of from the emitted C.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.cparser.includes import include_directives


# ---------------------------------------------------------------------------
# #include: the tree distinguishes a directive from text that looks like one
# ---------------------------------------------------------------------------

def test_a_commented_include_is_not_a_dependency() -> None:
    """A `#include` inside a comment is not a preprocessor directive.

    A line-anchored pattern collected it, so the cache depended on a header nothing
    included: editing that header invalidated a program that never read it. The
    grammar has no `preproc_include` node there, so there is nothing to collect.
    """
    source = (
        "// #include \"commented_out.h\"\n"
        "/* #include <also_commented.h> */\n"
        "#include <real.h>\n"
    )
    found = [name for name, _quoted in include_directives(source)]
    assert found == ["real.h"], found


def test_the_quoted_and_angled_forms_stay_distinguishable() -> None:
    """The two spellings search different paths, so the kind has to survive.

    `"..."` searches the including file's directory first; `<...>` searches only the
    include paths. The tree spells the difference as the child's node type
    (`string_literal` versus `system_lib_string`).
    """
    found = include_directives('#include "local.h"\n#include <system.h>\n')
    assert found == [("local.h", True), ("system.h", False)], found


def test_a_spaced_directive_needs_no_new_pattern() -> None:
    """`#  include` is the same directive, and the grammar says so.

    The pattern spelled the whitespace it happened to anticipate (`\\s*`), so this form
    worked -- but only because it was anticipated. The tree has no such notion: the
    directive is a node whatever the spacing between the `#` and the keyword.
    """
    found = include_directives("#  include <spaced.h>\n")
    assert found == [("spaced.h", False)], found


def test_the_include_order_is_preserved() -> None:
    """The recursion visits a nested include before its sibling, so order matters."""
    source = "#include <a.h>\n#include <b.h>\n#include <c.h>\n"
    found = [name for name, _quoted in include_directives(source)]
    assert found == ["a.h", "b.h", "c.h"], found


# ---------------------------------------------------------------------------
# range() step: the python node decides, not the emitted text
# ---------------------------------------------------------------------------

def _loop_header(kernel_object, backend: str = "opencl") -> str:
    """The generated `for` header of the loop **this** kernel compiled.

    Every kernel in one test module shares a generated source file, so taking the
    first `for (` in the file would answer whichever kernel happened to be written
    first. The kernel's own C name picks its definition apart.
    """
    compiled = next(iter(kernel_object._cache.values()))[0]
    c_name: str = compiled.name
    source = open(compiled._program.file_name, encoding="utf-8").read()
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if c_name + "(" in line and ("__kernel" in line or "kernel void" in line
                                     or "__global__" in line):
            for body_line in lines[index:]:
                if body_line.strip().startswith("for ("):
                    return body_line.strip()
    return ""


@kernel
def _count_down(n, out):
    """A descending loop: the step is a literal in the source."""
    for _i in range(n, 0, -1):
        out[0] = out[0] + 1


@kernel
def _count_up(n, out):
    """An ascending loop with a step above one."""
    for _i in range(0, n, 2):
        out[0] = out[0] + 1


@kernel
def _count_up_explicitly(n, out):
    """A unary `+` on the step is still a constant."""
    for _i in range(0, n, +3):
        out[0] = out[0] + 1


@kernel
def _count_by_a_runtime_step(n, step, out):
    """A step the device computes: there is no value for a C `for` header to carry."""
    for _i in range(0, n, step):
        out[0] = out[0] + 1


@pytest.mark.parametrize("kernel_object, op, emitted", [
    (_count_down, ">", "-1"),
    (_count_up, "<", "2"),
    (_count_up_explicitly, "<", "3"),
])
def test_a_constant_step_sets_the_loop_direction(gpu_backend, kernel_object, op,
                                                 emitted) -> None:
    """A negative step counts down, a positive one counts up.

    This is the case the old text pattern got wrong until it grew a parenthesis
    peeler: a negative literal is emitted as `(-1)`, so `-?\\d+` did not match it and
    every descending loop was refused with "step must be a constant integer". The
    node says `UnaryOp(USub, Constant(1))`, which needs no peeling.
    """
    out = np.zeros(1, np.int32)
    kernel_object[gpu_backend](4, out)
    header = _loop_header(kernel_object, gpu_backend)
    assert op in header, header
    assert "+= %s" % emitted in header, header


def test_a_dynamic_step_is_refused_by_name(gpu_backend) -> None:
    """A step the device computes has no value a C `for` header can carry.

    The refusal is the same one the text pattern gave; what changed is that it now
    comes from the node being a name rather than from its rendered text failing to
    look like digits.
    """
    with pytest.raises(SyntaxError) as caught:
        _count_by_a_runtime_step[gpu_backend](4, 2, np.zeros(1, np.int32))
    assert "constant integer" in str(caught.value)
