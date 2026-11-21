"""Reading a Metal source's declarations: structures, entry points, parameters.

The tree the Metal grammar produces is a C++ tree with MSL's words in it, so the
walk is a thin, explicit reader over that shape rather than a generic one. It
answers exactly what a launch and an imported module need:

* the entry points (`kernel void f(...)`) and each one's parameters,
* the `struct` declarations, with the ctypes layout `kernel_import` publishes,
* the per-parameter copy policy, which is what makes a read-only buffer skip its
  read-back.

Everything the grammar says is read from the node types -- no regular expression
touches the source text. That is the difference from the declaration scanner this
replaces: that one *found* declarations by pattern, so a spelling it did not
anticipate was silently missed. Here an unknown shape has a node type, and the
walk either understands it or refuses.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from tree_sitter import Node

from ....cparser.ts_parser.node_kinds import _STATEMENT_NODE_TYPES
from .node_kinds import (
    ADDRESS_SPACES,
    DECLARATOR_KINDS,
    DISPATCH_ATTRIBUTES,
    STAGE_WORDS,
    TYPE_KINDS,
)

if TYPE_CHECKING:
    from ....cparser.ts_parser import ParsedVar
    from ..program_info import ArgInfo, KernelInfo, StructInfo


class _MetalDeclarationsMixin:
    """The declaration half of the Metal walk."""

    def _text(self, node:Optional[Node])->str:
        """A node's source text, decoded."""
        return (node.text or b"").decode() if node is not None else ""

    # ---- structures ----

    def _parse_struct(self, node:Node)->None:
        """Record one `struct` declaration and its members."""
        name:str = ""
        body:Optional[Node] = None
        for child in node.named_children:
            if child.type == "type_identifier":
                name = self._text(child)
            elif child.type == "field_declaration_list":
                body = child
        if body is None or not name:
            # A forward declaration or an anonymous struct: there is no layout to build.
            return

        struct_info:"StructInfo" = self._new_struct_info(name, node.type == "union_specifier")
        self._struct_infos[name] = struct_info
        for field in body.named_children:
            if field.type != "field_declaration":
                continue
            head, declarators = self._declaration_head(field)
            for declarator in declarators:
                member = self._resolve_declarator(declarator, head)
                if member is None or not member["name"]:
                    continue
                struct_info.members[member["name"]] = self._new_var_info(member)

    # ---- functions ----

    def _parse_function(self, node:Node)->None:
        """Record one function definition, and whether it is a compute entry point."""
        stage:str = ""
        declarator:Optional[Node] = None
        for child in node.children:
            if child.type == "storage_class_specifier":
                word:str = self._text(child).strip()
                if word in STAGE_WORDS:
                    stage = word
            elif child.type == "function_declarator":
                declarator = child
        if declarator is None:
            return

        name:str = ""
        parameters:Optional[Node] = None
        for child in declarator.named_children:
            if child.type in ("identifier", "qualified_identifier"):
                name = self._text(child).split("::")[-1]
            elif child.type == "parameter_list":
                parameters = child
        if not name:
            return

        kernel_info:"KernelInfo" = self._new_kernel_info(name)
        if parameters is not None:
            for parameter in parameters.named_children:
                if parameter.type != "parameter_declaration":
                    continue
                self._add_parameter(kernel_info, parameter)

        self._function_infos[name] = kernel_info
        # The *body* is what the analysis walks. Handing it the whole definition would
        # make the parameter list a set of reads, and every argument look read.
        self._kernel_bodies[name] = next(
            (child for child in node.named_children
             if child.type == "compound_statement"),
            node)
        if stage == "kernel":
            self._kernel_infos[name] = kernel_info

    def _add_parameter(self, kernel_info:"KernelInfo", node:Node)->None:
        """One parameter declaration, including its dispatch attribute."""
        head, declarators = self._declaration_head(node)
        if not declarators:
            return
        desc = self._resolve_declarator(declarators[0], head)
        if desc is None or not desc["name"]:
            return

        attributes:List[str] = self._attribute_names(node)
        dispatch_supplied:bool = any(name in DISPATCH_ATTRIBUTES for name in attributes)

        arg_info:"ArgInfo" = self._new_arg_info(kernel_info, desc)
        # The attribute parameters are declared by the device signature but filled in by
        # the dispatch, so they are not part of the host-visible argument map. They are
        # recorded beside it because the launch has to skip those slots by index.
        arg_info.dispatch_supplied = dispatch_supplied
        if dispatch_supplied:
            self._dispatch_names.setdefault(kernel_info.name, []).append(arg_info.name)
            return
        kernel_info.args[arg_info.name] = arg_info

    def _attribute_names(self, node:Node)->List[str]:
        """Every `[[...]]` attribute's name under one declaration."""
        names:List[str] = []
        stack:List[Node] = [node]
        while stack:
            current = stack.pop()
            if current.type == "attribute_declaration":
                for child in current.named_children:
                    if child.type == "attribute":
                        names.append(self._text(child).strip())
                continue
            stack.extend(current.named_children)
        return names

    # ---- declarations ----

    def _declaration_head(self,
                          node:Node) -> Tuple["ParsedVar", List[Node]]:
        """Split a declaration into its shared head and its declarator nodes."""
        address:Optional[str] = None
        type_qualifiers:List[str] = []
        type_parts:List[str] = []
        declarators:List[Node] = []

        for child in node.children:
            kind:str = child.type
            if kind == "storage_class_specifier":
                word:str = self._text(child).strip()
                if word in ADDRESS_SPACES:
                    address = ADDRESS_SPACES[word]
            elif kind == "type_qualifier":
                type_qualifiers.append(self._text(child).strip())
            elif kind in TYPE_KINDS:
                type_parts.append(" ".join(self._text(child).split()))
            elif kind in DECLARATOR_KINDS:
                declarators.append(child)

        base:"ParsedVar" = {
            "address": address,
            "access": None,
            "type_qualifiers": type_qualifiers,
            "base_type": " ".join(type_parts),
        }
        return base, declarators

    def _resolve_declarator(self, node:Node,
                            base:"ParsedVar") -> Optional["ParsedVar"]:
        name, is_ptr, array_shape = self._walk_declarator(node)
        desc:"ParsedVar" = dict(base)  # type: ignore[assignment]
        desc["name"] = name
        desc["is_ptr"] = is_ptr
        desc["array_shape"] = array_shape
        desc["type_str"] = base["base_type"] + ("*" if is_ptr else "")
        desc["kernel_flags"] = self._flags_from_desc(desc)
        return desc

    def _walk_declarator(self, node:Node) -> Tuple[str, bool, Tuple[int, ...]]:
        """The name, pointer-ness and array extents a declarator spells."""
        kind:str = node.type
        if kind in ("identifier", "field_identifier"):
            return self._text(node), False, ()
        if kind == "pointer_declarator":
            for child in node.named_children:
                if child.type in DECLARATOR_KINDS:
                    name, _is_ptr, shape = self._walk_declarator(child)
                    return name, True, shape
            return "", True, ()
        if kind == "array_declarator":
            shape:Tuple[int, ...] = self._array_shape(node)
            for child in node.named_children:
                if child.type in DECLARATOR_KINDS:
                    name, is_ptr, _ = self._walk_declarator(child)
                    return name, is_ptr, shape
            return "", False, shape
        if kind in ("attributed_declarator", "parenthesized_declarator",
                    "init_declarator", "reference_declarator"):
            # `attributed_declarator` wraps the real declarator under the `[[...]]`
            # text: `gid [[thread_position_in_grid]]` is an identifier inside it.
            for child in node.named_children:
                if child.type in DECLARATOR_KINDS + ("reference_declarator",):
                    return self._walk_declarator(child)
            return "", False, ()
        if kind == "function_declarator":
            for child in node.named_children:
                if child.type in DECLARATOR_KINDS:
                    return self._walk_declarator(child)
            return "", False, ()
        return "", False, ()

    def _array_shape(self, node:Node) -> Tuple[int, ...]:
        """The constant extents an `array_declarator` declares.

        A C array parameter's first extent is optional (`float a[]`), and an extent
        may be any constant expression; only the numeric ones are read here, because
        the shape is the host-side layout and a symbolic extent has no host meaning.
        """
        extents:List[int] = []
        for child in node.named_children:
            if child.type not in ("array_declarator", "identifier"):
                text:str = self._text(child).strip()
                if text.isdigit():
                    extents.append(int(text))
        return tuple(extents)

    @staticmethod
    def _flags_from_desc(desc:"ParsedVar"):
        """The copy policy of one pointer argument.

        This is the coarse half of the policy -- what the *declaration* says. The body
        analysis in the analysis mixin narrows it further; the two are combined the way
        the C backends combine theirs, with a `const` qualifier authoritative.
        """
        from ....constants import KernelFlags

        flags = KernelFlags.Readed | KernelFlags.Writed
        if "const" in desc["type_qualifiers"] or desc["address"] == "constant":
            flags = KernelFlags.Readed
        return flags


# `_STATEMENT_NODE_TYPES` is re-exported so the analysis half can import it from one
# place, mirroring how the shared walk splits its node kinds by group.
__all__ = ["_MetalDeclarationsMixin", "_STATEMENT_NODE_TYPES"]

