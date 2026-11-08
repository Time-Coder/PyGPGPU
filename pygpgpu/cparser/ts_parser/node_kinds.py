"""The tree-sitter node kinds the walk recognizes.

The tuples are shared by the groups that make up the walk -- the declarations read the
type and declarator kinds, the body reads the statement kinds -- so they live here rather
than in whichever group happened to need them first.
"""
from __future__ import annotations

from typing import Tuple

_TYPE_NODE_TYPES:Tuple[str, ...] = ("scalar_type", "primitive_type", "sized_type_specifier", "other_builtin_type", "vector_type", "type_identifier", "enum_specifier")
_DECLARATOR_NODE_TYPES:Tuple[str, ...] = ("identifier", "field_identifier", "pointer_declarator", "array_declarator", "function_declarator", "parenthesized_declarator", "init_declarator")
_STATEMENT_NODE_TYPES:Tuple[str, ...] = (
    "compound_statement", "declaration", "expression_statement",
    "if_statement", "for_statement", "while_statement", "do_statement", "switch_statement",
    "return_statement", "break_statement", "continue_statement", "goto_statement",
    "case_statement", "labeled_statement"
)
