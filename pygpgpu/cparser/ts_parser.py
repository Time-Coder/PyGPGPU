from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from tree_sitter import Node

from ..constants import KernelFlags

if TYPE_CHECKING:
    from tree_sitter import Parser


_TYPE_NODE_TYPES:Tuple[str, ...] = ("scalar_type", "primitive_type", "sized_type_specifier", "other_builtin_type", "type_identifier", "enum_specifier")
_DECLARATOR_NODE_TYPES:Tuple[str, ...] = ("identifier", "field_identifier", "pointer_declarator", "array_declarator", "function_declarator", "parenthesized_declarator", "init_declarator")
_STATEMENT_NODE_TYPES:Tuple[str, ...] = (
    "compound_statement", "declaration", "expression_statement",
    "if_statement", "for_statement", "while_statement", "do_statement", "switch_statement",
    "return_statement", "break_statement", "continue_statement", "goto_statement",
    "case_statement", "labeled_statement"
)


class TSParserMixin:
    """Shared tree-sitter based parser for GPU kernel sources (OpenCL / CUDA).

    Host parser must provide:
        self._parser: tree_sitter.Parser
        self._clean_code: str
        self._struct_infos: Dict[str, StructInfo]
        self._kernel_infos: Dict[str, KernelInfo]
        _new_struct_info(name) -> StructInfo
        _new_var_info(desc) -> VarInfo
        _new_kernel_info(name) -> KernelInfo
        _new_arg_info(parent, desc) -> ArgInfo
        _is_mem_arg(arg) -> bool
    """

    _array_dim_pattern:re.Pattern = re.compile(r"\[\s*(\d+)\s*\]")

    _address_qualifier_names:Dict[str, str] = {
        "global": "global",
        "constant": "constant",
        "shared": "local",
        "local": "local",
        "private": "private"
    }
    _type_qualifier_names:Tuple[str, ...] = ("const", "volatile", "restrict", "pipe")
    _cuda_function_qualifiers:Tuple[str, ...] = ("__global__", "__device__", "__host__", "__forceinline__")

    def _ts_parse(self)->None:
        self._function_infos:Dict[str, Any] = {}
        self._kernel_bodies:Dict[str, Node] = {}
        self._helper_flags:Dict[str, Optional[Dict[str, KernelFlags]]] = {}
        self._analyzing_funcs:Set[str] = set()
        self._struct_aliases:Dict[str, str] = {}

        parse_code:str = (self._parse_code if getattr(self, "_parse_code", None) else self._clean_code)
        ast = self._parser.parse(parse_code.encode())
        for child in ast.root_node.named_children:
            if child.type == "type_definition":
                specifier, alias = child.named_children[0], (child.named_children[1].text.decode() if len(child.named_children) > 1 else "")
                if specifier.type in ("struct_specifier", "union_specifier"):
                    self._parse_struct(specifier, alias)
            elif child.type == "struct_specifier":
                self._parse_struct(child)
            elif child.type == "function_definition":
                self._parse_function(child)

        for alias_name, target_name in self._struct_aliases.items():
            target_name = self._struct_aliases.get(target_name, target_name)
            if target_name in self._struct_infos:
                self._struct_infos[alias_name] = self._struct_infos[target_name]

    def _parse_struct(self, node:Node, alias_name:str="")->None:
        struct_name:str = ""
        field_list:Optional[Node] = None
        for child in node.named_children:
            if child.type == "type_identifier":
                struct_name = child.text.decode()
            elif child.type == "field_declaration_list":
                field_list = child

        if field_list is None:
            used_name:str = (alias_name if alias_name else struct_name)
            if struct_name and used_name != struct_name:
                self._struct_aliases[used_name] = struct_name
            return

        used_name:str = (alias_name if alias_name else struct_name)
        struct_info:Any = self._new_struct_info(used_name)
        keys:Set[str] = {used_name}
        if struct_name:
            keys.add(struct_name)
            keys.add("struct " + struct_name)
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
                is_kernel = is_kernel or ("kernel" in child.text.decode()) or ("__global__" in child.text.decode())
            elif child.type == "__global__":
                is_kernel = True
            elif child.type == "function_declarator":
                declarator = child

        if declarator is None:
            return

        func_name:str = declarator.named_children[0].text.decode()
        kernel_info:Any = self._new_kernel_info(func_name)

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

    def _parse_declaration_head(self, node:Node)->Tuple[Dict[str, Any], List[Node]]:
        """Splits a (field|parameter|local) declaration into shared head info and declarator nodes."""
        address:Optional[str] = None
        access:Optional[str] = None
        type_qualifiers:List[str] = []
        type_parts:List[str] = []
        declarators:List[Node] = []

        for child in node.children:
            child_type:str = child.type
            if child_type == "address_space_qualifier":
                address = child.text.decode().strip("_")
            elif child_type == "access_qualifier":
                access = child.text.decode().strip("_")
            elif child_type == "type_qualifier":
                qualifier_name:str = child.text.decode().strip("_")
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
                qualifier_name:str = child.text.decode().strip("_")
                if qualifier_name in self._address_qualifier_names:
                    address = self._address_qualifier_names[qualifier_name]
                else:
                    type_parts.append(" ".join(child.text.decode().split()))
            elif child_type in _DECLARATOR_NODE_TYPES:
                declarators.append(child)

        base_desc:Dict[str, Any] = {
            "address": address,
            "access": access,
            "type_qualifiers": type_qualifiers,
            "base_type": " ".join(type_parts)
        }
        return base_desc, declarators

    def _resolve_declarator(self, node:Node, base_desc:Dict[str, Any])->Dict[str, Any]:
        name, is_ptr, array_shape = self._walk_declarator(node)
        desc:Dict[str, Any] = dict(base_desc)
        desc["name"] = name
        desc["is_ptr"] = is_ptr
        desc["array_shape"] = array_shape
        desc["type_str"] = base_desc["base_type"] + ("*" if is_ptr else "")
        desc["kernel_flags"] = self._flags_from_desc(desc)
        return desc

    def _walk_declarator(self, node:Node)->Tuple[str, bool, Tuple[int, ...]]:
        node_type:str = node.type
        if node_type in ("identifier", "field_identifier"):
            return node.text.decode(), False, ()

        if node_type == "pointer_declarator":
            for child in node.named_children:
                if child.type in _DECLARATOR_NODE_TYPES:
                    name, is_ptr, array_shape = self._walk_declarator(child)
                    return name, True, array_shape
            return "", True, ()

        if node_type == "array_declarator":
            dims:Tuple[int, ...] = tuple(int(x) for x in self._array_dim_pattern.findall(node.text.decode()))
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
                    return child.text.decode(), False, ()
            for child in node.named_children:
                if child.type in ("pointer_declarator", "parenthesized_declarator", "array_declarator"):
                    return self._walk_declarator(child)

        if node_type == "init_declarator":
            for child in node.named_children:
                if child.type in _DECLARATOR_NODE_TYPES:
                    return self._walk_declarator(child)

        return "", False, ()

    def _specifier_text(self, node:Node)->str:
        text:str = node.text.decode()
        brace_pos:int = text.find("{")
        if brace_pos != -1:
            text = text[:brace_pos]
        return " ".join(text.split())

    def _flags_from_desc(self, desc:Dict[str, Any])->KernelFlags:
        kernel_flags:KernelFlags = (KernelFlags.Readed | KernelFlags.Writed)
        if desc["access"] == "read_only" or desc["address"] == "constant" or "const" in desc["type_qualifiers"]:
            kernel_flags = KernelFlags.Readed
        elif desc["access"] == "write_only":
            kernel_flags = KernelFlags.Writed
        return kernel_flags

    # ---------------- kernel body read/write analysis ----------------

    def _analyze_kernels(self)->None:
        for name, kernel_info in self._kernel_infos.items():
            body:Optional[Node] = self._function_body(self._kernel_bodies.get(name))
            if body is None:
                continue
            self._analyze_kernel(kernel_info, body)

    def _analyze_kernel(self, kernel_info:Any, body:Node)->None:
        reads, writes, escapes, member_reads, member_writes = self._analyze_body(kernel_info, body)
        for name, flags in self._param_flags(kernel_info, reads, writes, escapes).items():
            kernel_info.args[name].kernel_flags = flags
        self._apply_member_flags(kernel_info, escapes, member_reads, member_writes)

    def _pointer_members(self, arg:Any)->Dict[str, Any]:
        """Pointer members of a struct-typed argument (empty for non-struct args)."""
        return {}

    def _apply_member_flags(self, kernel_info:Any, escapes:Set[str], member_reads:Dict[str, Set[str]], member_writes:Dict[str, Set[str]])->None:
        """Derives per-member KernelFlags for pointer members of by-value struct args.

        Flags are stored per-arg (arg.member_infos) so each kernel keeps its own
        copy policy; members never touched fall back to Readed|Writed.
        """
        for name, arg in kernel_info.args.items():
            members:Dict[str, Any] = self._pointer_members(arg)
            if not members:
                continue

            member_infos:Dict[str, Any] = {}
            escaped:bool = (name in escapes)
            for member_name, member in members.items():
                if escaped:
                    flags:KernelFlags = (KernelFlags.Readed | KernelFlags.Writed)
                else:
                    flags = ((KernelFlags.Readed if member_name in member_reads.get(name, set()) else KernelFlags(0)) |
                             (KernelFlags.Writed if member_name in member_writes.get(name, set()) else KernelFlags(0)))

                if not flags:
                    flags = (KernelFlags.Readed | KernelFlags.Writed)

                if member.readonly:
                    flags = KernelFlags.Readed
                elif member.writeonly:
                    flags = KernelFlags.Writed

                member_copy:Any = copy.copy(member)
                member_copy.kernel_flags = flags
                member_infos[member_name] = member_copy

            arg.member_infos = member_infos

    def _call_flags(self, func_name:str)->Optional[Dict[str, KernelFlags]]:
        """Per-parameter KernelFlags of a helper function, analyzed on demand."""
        if func_name in self._helper_flags:
            return self._helper_flags[func_name]

        func_info:Any = self._function_infos.get(func_name)
        if func_info is None or func_name in self._kernel_infos or func_name in self._analyzing_funcs:
            return None

        body:Optional[Node] = self._function_body(self._kernel_bodies.get(func_name))
        if body is None:
            return None

        self._analyzing_funcs.add(func_name)
        try:
            reads, writes, escapes, _, _ = self._analyze_body(func_info, body)
        finally:
            self._analyzing_funcs.discard(func_name)

        flags:Dict[str, KernelFlags] = self._param_flags(func_info, reads, writes, escapes)
        self._helper_flags[func_name] = flags
        return flags

    def _param_flags(self, info:Any, reads:Set[str], writes:Set[str], escapes:Set[str])->Dict[str, KernelFlags]:
        result:Dict[str, KernelFlags] = {}
        for name, arg in info.args.items():
            if not self._is_mem_arg(arg):
                result[name] = KernelFlags.Readed
                continue

            if name in escapes:
                flags:KernelFlags = (KernelFlags.Readed | KernelFlags.Writed)
            else:
                flags = ((KernelFlags.Readed if name in reads else KernelFlags(0)) | (KernelFlags.Writed if name in writes else KernelFlags(0)))

            if not flags:
                flags = (KernelFlags.Readed | KernelFlags.Writed)

            if arg.readonly:
                flags = KernelFlags.Readed
            elif arg.writeonly:
                flags = KernelFlags.Writed

            result[name] = flags
        return result

    def _function_body(self, node:Optional[Node])->Optional[Node]:
        if node is None or node.type != "function_definition":
            return None
        return next((c for c in node.named_children if c.type == "compound_statement"), None)

    def _analyze_body(self, info:Any, body:Node)->Tuple[Set[str], Set[str], Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
        ctx:Dict[str, Any] = {
            "reads": set(), "writes": set(), "escapes": set(),
            "member_reads": {}, "member_writes": {}
        }
        roots:Dict[str, Tuple[Optional[str], bool]] = {}
        for name, arg in info.args.items():
            roots[name] = (name, self._is_mem_arg(arg))
        ctx["struct_args"] = {name for name, arg in info.args.items() if self._pointer_members(arg)}

        scopes:List[Dict[str, Tuple[Optional[str], bool]]] = [roots]
        for child in body.named_children:
            self._visit_statement(child, scopes, ctx)

        return ctx["reads"], ctx["writes"], ctx["escapes"], ctx["member_reads"], ctx["member_writes"]

    def _resolve(self, name:str, scopes:List[Dict[str, Tuple[Optional[str], bool]]])->Tuple[Optional[str], bool]:
        for scope in reversed(scopes):
            if name in scope:
                return scope[name]
        return None, False

    def _resolve_expr_root(self, node:Node, scopes:List[Dict[str, Tuple[Optional[str], bool]]])->Tuple[Optional[str], bool]:
        if node.type in ("identifier", "field_identifier"):
            return self._resolve(node.text.decode(), scopes)
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
            root, is_ptr = self._resolve(node.text.decode(), scopes)
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

    def _collect_root_refs(self, node:Node, scopes, found:Set[str])->None:
        """Collects every mem-arg root referenced inside an expression (for conservative escapes)."""
        root, is_mem = self._resolve_expr_root(node, scopes)
        if root is not None and is_mem:
            found.add(root)
            return

        for child in node.named_children:
            self._collect_root_refs(child, scopes, found)

    def _visit_expression(self, node:Node, scopes, ctx, is_write:bool)->None:
        node_type:str = node.type

        if node_type in ("identifier", "field_identifier"):
            root, _ = self._resolve(node.text.decode(), scopes)
            if root is not None:
                (ctx["writes"] if is_write else ctx["reads"]).add(root)
            return

        if node_type == "subscript_expression":
            children:List[Node] = node.named_children
            base:Node = children[0]
            root, is_ptr = self._resolve_expr_root(base, scopes)
            if root is not None and is_ptr:
                (ctx["writes"] if is_write else ctx["reads"]).add(root)
                if base.type == "subscript_expression":
                    self._visit_expression(base, scopes, ctx, is_write)
            else:
                self._visit_expression(base, scopes, ctx, is_write)
            for index in children[1:]:
                self._visit_expression(index, scopes, ctx, False)
            return

        if node_type == "pointer_expression":
            operator:str = (node.text.decode()[0] if node.text else "*")
            operand:Optional[Node] = (node.named_children[0] if node.named_children else None)
            if operand is None:
                return
            if operator == "&":
                root, _ = self._resolve_expr_root(operand, scopes)
                if root is not None:
                    ctx["escapes"].add(root)
                else:
                    self._visit_expression(operand, scopes, ctx, False)
            else:
                root, is_ptr = self._resolve_expr_root(operand, scopes)
                if root is not None and is_ptr:
                    (ctx["writes"] if is_write else ctx["reads"]).add(root)
                else:
                    self._visit_expression(operand, scopes, ctx, is_write)
            return

        if node_type in ("field_expression", "member_access_expression"):
            base:Node = node.named_children[0]
            root, is_ptr = self._resolve_expr_root(base, scopes)
            if root is not None:
                (ctx["writes"] if is_write else ctx["reads"]).add(root)

                field:Node = node.named_children[-1]
                if field.type in ("field_identifier", "identifier"):
                    member_bucket:Dict[str, Set[str]] = (ctx["member_writes"] if is_write else ctx["member_reads"])
                    member_bucket.setdefault(root, set()).add(field.text.decode())

                if base.type != "identifier":
                    self._visit_expression(base, scopes, ctx, False)
            else:
                self._visit_expression(base, scopes, ctx, is_write)
            return

        if node_type == "assignment_expression":
            left:Node = node.named_children[0]
            right:Node = node.named_children[-1]
            operator:str = "="
            for child in node.children:
                if (not child.is_named) and child.text and child.type not in ("(", ")"):
                    operator = child.text.decode()
                    break

            lroot, lisptr = self._resolve_expr_root(left, scopes)
            if lisptr and left.type == "identifier":
                source:Optional[str] = self._alias_source(right, scopes)
                if source is not None:
                    self._bind_symbol(left.text.decode(), (source, True), scopes)
                    return

                found:Set[str] = set()
                self._collect_root_refs(right, scopes, found)
                ctx["escapes"].update(found)

            self._visit_expression(left, scopes, ctx, True)
            if operator != "=":
                self._visit_expression(left, scopes, ctx, False)
            self._visit_expression(right, scopes, ctx, False)
            return

        if node_type == "update_expression":
            operand:Node = node.named_children[0]
            root, is_ptr = self._resolve_expr_root(operand, scopes)
            if root is not None and is_ptr and operand.type == "identifier":
                return
            self._visit_expression(operand, scopes, ctx, True)
            self._visit_expression(operand, scopes, ctx, False)
            return

        if node_type == "call_expression":
            self._visit_call(node, scopes, ctx)
            return

        for child in node.named_children:
            self._visit_expression(child, scopes, ctx, False)

    def _visit_call(self, node:Node, scopes, ctx)->None:
        children:List[Node] = node.named_children
        func:Node = children[0]
        func_name:str = (func.text.decode() if func.type == "identifier" else "")
        arg_list:Node = (children[1] if len(children) > 1 and children[1].type == "argument_list" else None)
        actuals:List[Node] = (list(arg_list.named_children) if arg_list is not None else [])

        helper_flags:Optional[Dict[str, KernelFlags]] = None
        helper_info:Any = None
        if func_name:
            helper_info = self._function_infos.get(func_name)
            helper_flags = self._call_flags(func_name)
        param_names:List[str] = (list(helper_flags.keys()) if helper_flags is not None else [])

        for i, actual in enumerate(actuals):
            root, is_mem = self._resolve_expr_root(actual, scopes)

            if helper_flags is not None and i < len(param_names):
                flags:KernelFlags = helper_flags[param_names[i]]
                param_is_mem:bool = (helper_info is not None and self._is_mem_arg(helper_info.args[param_names[i]]))

                if root is not None and is_mem and param_is_mem:
                    # 直接传指针本身:不构成数据访问,只传播 helper 的读写集
                    if flags & KernelFlags.Readed:
                        ctx["reads"].add(root)
                    if flags & KernelFlags.Writed:
                        ctx["writes"].add(root)
                    continue

                self._visit_expression(actual, scopes, ctx, False)
                target:Optional[str] = root
                if target is None:
                    target = self._alias_source(actual, scopes)
                if target is not None:
                    if flags & KernelFlags.Readed:
                        ctx["reads"].add(target)
                    if flags & KernelFlags.Writed:
                        ctx["writes"].add(target)
            else:
                self._visit_expression(actual, scopes, ctx, False)
                target = (root if (root is not None and (is_mem or root in ctx["struct_args"])) else self._alias_source(actual, scopes))
                if target is not None:
                    ctx["escapes"].add(target)
