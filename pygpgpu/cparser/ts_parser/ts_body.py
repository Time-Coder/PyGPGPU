# The body of a function: the statements, the symbols they bind, and the
# aliases a declaration creates.
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
from tree_sitter import Node

from .node_kinds import (
    _STATEMENT_NODE_TYPES,
)


class _TsBodyMixin:

    def _resolve(self, name:str, scopes:List[Dict[str, Tuple[Optional[str], bool]]])->Tuple[Optional[str], bool]:
        for scope in reversed(scopes):
            if name in scope:
                return scope[name]
        return None, False


    def _resolve_expr_root(self, node:Node, scopes:List[Dict[str, Tuple[Optional[str], bool]]])->Tuple[Optional[str], bool]:
        if node.type in ("identifier", "field_identifier"):
            return self._resolve(self._node_text(node), scopes)
        return None, False


    def _bind_symbol(self, name:str, symbol:Tuple[Optional[str], bool], scopes:List[Dict[str, Tuple[Optional[str], bool]]])->None:
        for scope in reversed(scopes):
            if name in scope:
                scope[name] = symbol
                return
        scopes[-1][name] = symbol


    def _visit_statement(self, node:Node, scopes, ctx)->None:
        node_type:str = node.type
        if node_type == "compound_statement":
            scopes.append({})
            for child in node.named_children:
                self._visit_statement(child, scopes, ctx)
            scopes.pop()
        elif node_type == "declaration":
            self._handle_declaration(node, scopes, ctx)
        elif node_type in _STATEMENT_NODE_TYPES:
            for child in node.named_children:
                if child.type in _STATEMENT_NODE_TYPES or child.type == "compound_statement":
                    self._visit_statement(child, scopes, ctx)
                else:
                    self._visit_expression(child, scopes, ctx, False)
        else:
            for child in node.named_children:
                self._visit_expression(child, scopes, ctx, False)


    def _handle_declaration(self, node:Node, scopes, ctx)->None:
        base_desc, declarators = self._parse_declaration_head(node)
        for declarator in declarators:
            if declarator.type == "init_declarator":
                inner:Node = declarator.named_children[0]
                inits:List[Node] = list(declarator.named_children[1:])
            else:
                inner, inits = declarator, []

            name, is_ptr, _ = self._walk_declarator(inner)
            if not name:
                continue

            root:Optional[str] = None
            if is_ptr and inits:
                source:Optional[str] = self._alias_source(inits[0], scopes)
                if source is not None:
                    root = source
                else:
                    for value in inits:
                        self._visit_expression(value, scopes, ctx, False)
                    found:Set[str] = set()
                    for value in inits:
                        self._collect_root_refs(value, scopes, found)
                    ctx["escapes"].update(found)
            else:
                for value in inits:
                    self._visit_expression(value, scopes, ctx, False)

            scopes[-1][name] = (root, is_ptr)


    def _alias_source(self, node:Node, scopes)->Optional[str]:
        """Returns the root arg name if node is a pure pointer-value expression (alias binding)."""
        node_type:str = node.type
        if node_type == "parenthesized_expression":
            inner:List[Node] = node.named_children
            if inner:
                return self._alias_source(inner[0], scopes)
            return None

        if node_type == "identifier" or node_type == "field_identifier":
            root, is_ptr = self._resolve(self._node_text(node), scopes)
            return root if (root is not None and is_ptr) else None

        if node_type == "binary_expression":
            left, right = (node.named_children + [None, None])[:2]
            if left is None or right is None:
                return None
            lroot, lptr = self._resolve_expr_root(left, scopes)
            rroot, rptr = self._resolve_expr_root(right, scopes)
            if lptr and not rptr:
                return lroot
            if rptr and not lptr:
                return rroot
            return None

        return None


    def _collect_addr_taken_roots(self, node:Node, scopes, found:Set[str])->None:
        """Collect array roots in an argument expression exposed via address-of ('&' form only; passing an element by value does not count)."""
        if node.type == "pointer_expression" and node.text and self._node_text(node).lstrip().startswith("&"):
            operand:Optional[Node] = (node.named_children[0] if node.named_children else None)
            if operand is not None:
                root, is_mem = self._resolve_expr_root(operand, scopes)
                if root is None and operand.type == "subscript_expression":
                    base:Optional[Node] = (operand.named_children[0] if operand.named_children else None)
                    if base is not None:
                        root, is_mem = self._resolve_expr_root(base, scopes)
                if root is not None and is_mem:
                    found.add(root)
                    return
        for child in node.named_children:
            self._collect_addr_taken_roots(child, scopes, found)


    def _collect_root_refs(self, node:Node, scopes, found:Set[str])->None:
        """Collects every mem-arg root referenced inside an expression (for conservative escapes)."""
        root, is_mem = self._resolve_expr_root(node, scopes)
        if root is not None and is_mem:
            found.add(root)
            return

        for child in node.named_children:
            self._collect_root_refs(child, scopes, found)
