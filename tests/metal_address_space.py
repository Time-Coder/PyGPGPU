"""Check the address-space qualifiers in a generated MSL source.

`metal_syntax_check` defines `device`/`threadgroup`/`constant` away so a C++ compiler
can read the rest of the source, which means a buffer declared in the *wrong* address
space passes it silently. This module checks the qualifiers as text instead, from the
one fact that matters at a kernel boundary.

MSL requires a pointer parameter to name its space, because the address space decides
which memory the pointer reaches and what the host may bind to it. The two spellings
this backend emits are:

* ``device`` for every caller buffer -- the arrays, the returned-value slots, the
  return claim/site words, the outer-scope arrays, the host-created ones;
* ``threadgroup`` for a `local_array_per_group` scratch buffer, which the dispatch gives
  each group privately.

A pointer parameter with **no** qualifier is the failure this catches: it is what an
OpenCL-shaped emission produces when the `_buffer_qualifier()` call is missed, and
Metal rejects it at compile time with a message about a missing address space rather
than naming the parameter. The check is therefore about the *presence* of a
qualifier, not about which one -- the choice between `device` and `threadgroup`
depends on what the buffer is, which the caller states.
"""
from __future__ import annotations

import re
from typing import List, NamedTuple

ADDRESS_SPACES: frozenset = frozenset(("device", "constant", "threadgroup"))

# One kernel declaration, so the parameter list can be split without leaving the
# function it belongs to.
_KERNEL_RE: re.Pattern = re.compile(r"kernel\s+void\s+\w+\s*\(([^)]*)\)", re.DOTALL)

# A parameter naming a pointer, with the leading qualifier tokens it carries.
_POINTER_RE: re.Pattern = re.compile(r"([^,]*?)\b\w+\s*(?:\[[^\]]*\])?\s*$")


class PointerParameter(NamedTuple):
    """One pointer parameter: its declaration text and, if any, its address space."""

    declaration: str
    address_space: str


def pointer_parameters(source: str) -> List[PointerParameter]:
    """Every pointer parameter of every kernel in `source`."""
    found: List[PointerParameter] = []
    for match in _KERNEL_RE.finditer(source):
        for raw in match.group(1).split(","):
            declaration: str = raw.strip()
            # Strip a trailing attribute (`[[thread_position_in_grid]]`) before looking
            # for the pointer, since that text is not part of the declaration's type.
            declaration = re.sub(r"\[\[[^\]]*\]\]", "", declaration).strip()
            if "*" not in declaration:
                continue
            space: str = ""
            for token in declaration.split():
                if token in ADDRESS_SPACES:
                    space = token
                    break
            found.append(PointerParameter(declaration, space))
    return found


def check_address_spaces(source: str) -> List[str]:
    """The pointer parameters of `source` that name no address space.

    An empty list means every pointer parameter is spelled `device ...` or
    `threadgroup ...`, which is what MSL requires.
    """
    problems: List[str] = []
    for parameter in pointer_parameters(source):
        if not parameter.address_space:
            problems.append(
                f"a pointer parameter names no address space: {parameter.declaration}")
    return problems
