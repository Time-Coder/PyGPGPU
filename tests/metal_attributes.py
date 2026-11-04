"""Check the attributes in a generated MSL source, which the C++ check strips.

`metal_syntax_check` defines the `[[...]]` attributes away so a C++ compiler can read
the rest, which means a *misspelled* or *misplaced* attribute passes it silently and
only the Metal compiler would object. This module closes that gap from the other
side: it reads the attributes as text and checks the three facts that decide whether
Metal accepts them.

1. **Every attribute name is one MSL declares.** A typo (`thread_position_in_grid_`)
   is a compiler error naming an unknown attribute, and the name is the whole
   contract between the kernel signature and the dispatch.
2. **The attribute names a dispatch-supplied value, not a host binding.** MSL's
   `[[buffer(n)]]`, `[[texture(n)]]` and `[[sampler(n)]]` are how the *host* binds a
   argument; a dispatch attribute is filled in by the threadgroup dispatch. Mixing
   them up would make the launch bind a buffer where the GPU writes a thread id.
3. **It sits after the parameter's name, on a parameter of the right shape.** MSL
   spells an attributed parameter `type name [[attr]]`; an attribute placed after
   the type (`float [[attr]] v`) is not what it looks like.

The set of names below is MSL's own list for compute kernels (Metal Shading
Language 3.x, "Kernel Function Arguments" / "Built-in Variables"). It is written out
rather than imported so that adding a new attribute to the translator is a visible
edit here too -- an unknown name in a generated source should be a test failure, not
something the checker waves through.
"""
from __future__ import annotations

import re
from typing import List, NamedTuple, Optional


# The attributes MSL declares for a compute kernel's arguments. Split by what they
# name: the thread/workgroup position family, the geometry of the dispatch, and the
# SIMD/quadgroup ones.
DISPATCH_ATTRIBUTES: frozenset = frozenset((
    "thread_position_in_grid",
    "thread_position_in_threadgroup",
    "thread_index_in_threadgroup",
    "threadgroup_position_in_grid",
    "threadgroups_per_grid",
    "threads_per_grid",
    "threads_per_threadgroup",
    "thread_index_in_threadgroup_imageblock",
    "thread_position_in_threadgroup_imageblock",
    "threadgroup_imageblock_position_in_grid",
    "quadgroup_index_in_threadgroup",
    "quadgroups_per_threadgroup",
    "simdgroup_index_in_threadgroup",
    "simdgroups_per_threadgroup",
    "thread_index_in_simdgroup",
    "threads_per_simdgroup",
    "simdgroup_count",
    "thread_index_in_quadgroup",
    "threads_per_quadgroup",
))

# The host-binding attributes: these are how a launch supplies its own arguments.
BINDING_ATTRIBUTES: frozenset = frozenset(("buffer", "texture", "sampler", "threadgroup"))

# A parameter declaration carrying one or more attributes. The declaration runs from
# the previous `,`/`(` to the next comma, so the parameter's own text is bounded.
_ATTRIBUTE_RE: re.Pattern = re.compile(r"\[\[\s*([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*\]\]")


class Attribute(NamedTuple):
    """One `[[name]]`/`[[name(arg)]]` occurrence with the text it belongs to."""

    name: str
    argument: Optional[str]
    declaration: str


def attributes_of(source: str) -> List[Attribute]:
    """Every attribute in `source`, with the parameter declaration it sits in."""
    found: List[Attribute] = []
    for match in _ATTRIBUTE_RE.finditer(source):
        name, argument = match.group(1), match.group(2)
        # The enclosing parameter declaration: from the nearest delimiter that starts
        # a declarator (`,` or `(`) up to the enclosing `)`.
        start: int = max(source.rfind(",", 0, match.start()),
                         source.rfind("(", 0, match.start())) + 1
        end: int = source.find(")", match.end())
        declaration: str = source[start:end if end != -1 else len(source)].strip()
        found.append(Attribute(name, argument, declaration))
    return found


def check_attributes(source: str) -> List[str]:
    """The problems in `source`'s attributes, as human-readable lines.

    An empty list means every attribute is one MSL declares, names a dispatch value
    rather than a host binding, and sits on a parameter declaration that ends with
    it.
    """
    problems: List[str] = []
    for attribute in attributes_of(source):
        name: str = attribute.name
        if name == "threadgroup":
            # The threadgroup-memory attribute is legitimate and this package does
            # emit it (a `zeros_per_group` buffer becomes `threadgroup uint*`).
            continue
        if name in BINDING_ATTRIBUTES:
            # `[[buffer(n)]]`-style attributes belong to the host's own arguments; the
            # generated kernels never declare one, because the host binds by index.
            problems.append(
                f"[[{name}]] is a host-binding attribute, not a dispatch value: "
                f"{attribute.declaration}")
            continue
        if name not in DISPATCH_ATTRIBUTES:
            problems.append(
                f"[[{name}]] is not an MSL dispatch attribute: {attribute.declaration}")
            continue
        # The attribute must end its declaration: MSL spells `type name [[attr]]`.
        trailing: str = attribute.declaration
        if not trailing.rstrip().endswith("]]"):
            problems.append(
                f"[[{name}]] must end its parameter declaration: {trailing}")
    return problems
