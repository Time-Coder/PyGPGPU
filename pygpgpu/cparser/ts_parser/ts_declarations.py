# The declarations of a C source: the structs, the functions and the
# declarators behind them, and the qualifier tables they are read from.
from __future__ import annotations
from typing import TYPE_CHECKING
import copy
from typing import List, Optional, Set, Tuple
from tree_sitter import Node
from ...constants import KernelFlags


if TYPE_CHECKING:
    from . import (
        ParsedVar,
        StructInfoLike,
        KernelInfoLike,
    )

from .node_kinds import (
    _TYPE_NODE_TYPES,
    _DECLARATOR_NODE_TYPES,
)


class _TsDeclarationsMixin:

    def _parse_struct(self, node:Node, alias_name:str="")->None:
        struct_name:str = ""
        field_list:Optional[Node] = None
        for child in node.named_children:
            if child.type == "type_identifier":
                struct_name = self._node_text(child)
            elif child.type == "field_declaration_list":
                field_list = child

        if field_list is None:
            used_name:str = (alias_name if alias_name else struct_name)
            if struct_name and used_name != struct_name:
                self._struct_aliases[used_name] = struct_name
            return

        used_name:str = (alias_name if alias_name else struct_name)
        # A union and a structure parse identically -- the members are the same list
        # -- but they lay out differently: a union gives every member the same address.
        # The node that named this declaration is the only record of which one it is.
        is_union:bool = node.type == "union_specifier"
        struct_info:StructInfoLike = self._new_struct_info(used_name, is_union)
        keys:Set[str] = {used_name}
        if struct_name:
            keys.add(struct_name)
            keys.add("struct " + struct_name)
            # A union's name is spelled with its own keyword, exactly as a structure's is:
            # a declaration written `union Tag { ... };` names its type `union Tag`, and a
            # member declared that way reaches the parser's type text with the keyword in
            # it. Registering only `struct Tag` left a generated `union U_<hash>` unreachable
            # by the member declarations that name it.
            keys.add("union " + struct_name)
        for key in keys:
            self._struct_infos[key] = struct_info

        for field_declaration in field_list.named_children:
            if field_declaration.type != "field_declaration":
                continue

            base_desc, declarators = self._parse_declaration_head(field_declaration)
            for declarator in declarators:
                if declarator.type == "init_declarator":
                    continue

                desc = self._resolve_declarator(declarator, base_desc)
                if not desc["name"]:
                    continue

                struct_info.members[desc["name"]] = self._new_var_info(desc)


    def _parse_function(self, node:Node)->None:
        is_kernel:bool = False
        declarator:Optional[Node] = None
        for child in node.children:
            if child.type == "function_qualifier":
                is_kernel = is_kernel or ("kernel" in self._node_text(child)) or ("__global__" in self._node_text(child))
            elif child.type == "__global__":
                is_kernel = True
            elif child.type == "function_declarator":
                declarator = child

        if declarator is None:
            return

        func_name:str = self._node_text(declarator.named_children[0])
        kernel_info:KernelInfoLike = self._new_kernel_info(func_name)

        param_list:Optional[Node] = next((c for c in declarator.named_children if c.type == "parameter_list"), None)
        if param_list is not None:
            for param in param_list.named_children:
                if param.type != "parameter_declaration":
                    continue

                base_desc, declarators = self._parse_declaration_head(param)
                if not declarators:
                    continue

                desc = self._resolve_declarator(declarators[0], base_desc)
                if not desc["name"]:
                    continue

                kernel_info.args[desc["name"]] = self._new_arg_info(kernel_info, desc)

        self._function_infos[func_name] = kernel_info
        self._kernel_bodies[func_name] = declarator.parent
        if is_kernel:
            self._kernel_infos[func_name] = kernel_info


    def _parse_declaration_head(self, node:Node)->Tuple[ParsedVar, List[Node]]:
        """Splits a (field|parameter|local) declaration into shared head info and declarator nodes."""
        address:Optional[str] = None
        access:Optional[str] = None
        type_qualifiers:List[str] = []
        type_parts:List[str] = []
        declarators:List[Node] = []

        for child in node.children:
            child_type:str = child.type
            if child_type == "address_space_qualifier":
                address = self._node_text(child).strip("_")
            elif child_type == "access_qualifier":
                access = self._node_text(child).strip("_")
            elif child_type == "type_qualifier":
                qualifier_name:str = self._node_text(child).strip("_")
                if qualifier_name in self._address_qualifier_names:
                    address = self._address_qualifier_names[qualifier_name]
                else:
                    type_qualifiers.append(qualifier_name)
            elif child_type in self._cuda_function_qualifiers or child_type == "__shared__":
                qualifier_name:str = child_type.strip("_")
                if qualifier_name in self._address_qualifier_names:
                    address = self._address_qualifier_names[qualifier_name]
            elif child_type == "struct_specifier" or child_type == "union_specifier":
                type_parts.append(self._specifier_text(child))
            elif child_type in _TYPE_NODE_TYPES:
                qualifier_name:str = self._node_text(child).strip("_")
                if qualifier_name in self._address_qualifier_names:
                    address = self._address_qualifier_names[qualifier_name]
                else:
                    type_parts.append(" ".join(self._node_text(child).split()))
            elif child_type in _DECLARATOR_NODE_TYPES:
                declarators.append(child)

        base_desc:ParsedVar = {
            "address": address,
            "access": access,
            "type_qualifiers": type_qualifiers,
            "base_type": " ".join(type_parts)
        }
        return base_desc, declarators


    def _declared_extents(self, node:Node)->Tuple[int, ...]:
        """The constant extents an `array_declarator` declares, outermost first.

        The grammar nests one `array_declarator` per dimension, so `a[4][8]` is an
        outer node whose extent (8) is a `number_literal` child and whose *first*
        child is the inner `a[4]`. Reading the literals in tree order is what puts
        the extents back in the order the declaration wrote them.

        This used to be a regular expression over the declarator's text, which read
        the right numbers but could not tell an extent from a digit inside anything
        else it happened to match. The tree has the extent as its own node, so a
        symbolic or absent extent (`a[]`, `a[N]`) is now simply not a
        `number_literal` rather than a pattern that failed to match.
        """
        extents:List[int] = []
        for child in node.children:
            if child.type == "array_declarator":
                extents.extend(self._declared_extents(child))
                break
            if child.type == "number_literal":
                text:str = self._node_text(child).strip().replace("'", "")
                if text.isdigit():
                    extents.append(int(text))
        return tuple(extents)

    def _walk_declarator(self, node:Node)->Tuple[str, bool, Tuple[int, ...]]:
        node_type:str = node.type
        if node_type in ("identifier", "field_identifier"):
            return self._node_text(node), False, ()

        if node_type == "pointer_declarator":
            for child in node.named_children:
                if child.type in _DECLARATOR_NODE_TYPES:
                    name, is_ptr, array_shape = self._walk_declarator(child)
                    return name, True, array_shape
            return "", True, ()

        if node_type == "array_declarator":
            dims:Tuple[int, ...] = self._declared_extents(node)
            for child in node.named_children:
                if child.type in _DECLARATOR_NODE_TYPES:
                    name, is_ptr, _ = self._walk_declarator(child)
                    return name, is_ptr, dims
            return "", False, dims

        if node_type == "parenthesized_declarator":
            for child in node.named_children:
                if child.type in _DECLARATOR_NODE_TYPES:
                    return self._walk_declarator(child)

        if node_type == "function_declarator":
            for child in node.named_children:
                if child.type == "identifier":
                    return self._node_text(child), False, ()
            for child in node.named_children:
                if child.type in ("pointer_declarator", "parenthesized_declarator", "array_declarator"):
                    return self._walk_declarator(child)

        if node_type == "init_declarator":
            for child in node.named_children:
                if child.type in _DECLARATOR_NODE_TYPES:
                    return self._walk_declarator(child)

        return "", False, ()


    def _resolve_declarator(self, node:Node, base_desc:ParsedVar)->ParsedVar:
        name, is_ptr, array_shape = self._walk_declarator(node)
        desc:ParsedVar = copy.deepcopy(base_desc)
        desc["name"] = name
        desc["is_ptr"] = is_ptr
        desc["array_shape"] = array_shape
        desc["type_str"] = base_desc["base_type"] + ("*" if is_ptr else "")
        desc["kernel_flags"] = self._flags_from_desc(desc)
        return desc


    def _specifier_text(self, node:Node)->str:
        text:str = self._node_text(node)
        brace_pos:int = text.find("{")
        if brace_pos != -1:
            text = text[:brace_pos]
        return " ".join(text.split())


    def _flags_from_desc(self, desc:ParsedVar)->KernelFlags:
        kernel_flags:KernelFlags = (KernelFlags.Readed | KernelFlags.Writed)
        if desc["access"] == "read_only" or desc["address"] == "constant" or "const" in desc["type_qualifiers"]:
            kernel_flags = KernelFlags.Readed
        elif desc["access"] == "write_only":
            kernel_flags = KernelFlags.Writed
        return kernel_flags
