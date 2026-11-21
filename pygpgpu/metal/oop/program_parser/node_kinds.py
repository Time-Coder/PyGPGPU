"""The node kinds the Metal grammar produces that the shared walk does not know.

The Metal grammar descends from tree-sitter-cpp, so most node kinds are the C/C++
ones the shared walk already recognizes. Three groups differ and are listed here:

* the *address space* words MSL writes as a storage class (`device`, `constant`,
  `threadgroup`, `thread`, ...), which the walk must treat as an address rather
  than as part of a type;
* the *dispatch attributes* (`[[thread_position_in_grid]]` and friends), which mark
  a parameter the threadgroup dispatch fills in rather than one the host binds;
* the declarator shape the grammar gives an attributed parameter, which wraps the
  ordinary declarator instead of replacing it.

Keeping them in one module means the walk has exactly one place to look for "a
kind that is Metal's".
"""
from __future__ import annotations

from typing import Dict, Tuple

# The MSL address spaces, as the grammar spells them in `storage_class_specifier`.
# Each maps to the normalized name the rest of the package uses, which follows
# OpenCL's spellings because the translator already speaks them (`__global` is a
# buffer, `__local` is group memory).
ADDRESS_SPACES:Dict[str, str] = {
    "device": "global",
    "constant": "constant",
    "threadgroup": "local",
    "threadgroup_imageblock": "local",
    "thread": "private",
    "ray_data": "global",
    "object_data": "global",
}

# The words that select a shader stage. `kernel` is the compute one this package
# launches; the other two are recorded so a source that declares them is not
# mistaken for a compute entry point.
STAGE_WORDS:frozenset = frozenset((
    "kernel", "vertex", "fragment", "visible",
))

# The attributes that name a *dispatch-supplied* value. A parameter carrying one is
# filled in by the threadgroup dispatch, so the host binds nothing for it -- the
# same rule Metal's declaration scanner applied before the grammar existed.
DISPATCH_ATTRIBUTES:frozenset = frozenset((
    "thread_position_in_grid", "thread_position_in_threadgroup",
    "thread_index_in_threadgroup", "threadgroup_position_in_grid",
    "threadgroups_per_grid", "threads_per_grid", "threads_per_threadgroup",
    "thread_index_in_threadgroup_imageblock",
    "thread_position_in_threadgroup_imageblock",
    "quadgroup_index_in_threadgroup", "simdgroup_index_in_threadgroup",
    "simdgroup_count", "thread_index_in_simdgroup", "threads_per_simdgroup",
))

# The declarator kinds the walk descends through to reach a name.
DECLARATOR_KINDS:Tuple[str, ...] = (
    "identifier", "field_identifier", "pointer_declarator", "array_declarator",
    "function_declarator", "parenthesized_declarator", "init_declarator",
    "attributed_declarator",
    # A reference parameter (`constant float& alpha`, what the generated kernels
    # declare for every scalar) is a declarator too: without it here, the whole
    # parameter is skipped and the launch reads a shorter signature than the
    # device one -- a scalar bound to the wrong slot, or "takes 3 arguments but 5
    # were given".
    "reference_declarator",
)

# The type kinds the walk reads as the head of a declaration.
TYPE_KINDS:Tuple[str, ...] = (
    "primitive_type", "type_identifier", "sized_type_specifier", "struct_specifier",
    "union_specifier",
    "enum_specifier", "template_type", "dependent_type", "decltype",
)

# Qualifiers that describe how a value may be accessed, not where it lives.
ACCESS_QUALIFIERS:frozenset = frozenset((
    "const", "volatile", "restrict", "__restrict", "mutable", "constinit", "consteval",
))
