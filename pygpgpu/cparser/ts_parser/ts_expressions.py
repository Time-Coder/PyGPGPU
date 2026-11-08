# The expressions of a body: what a call passes, and what a type descriptor
# spells.
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Dict, List, Optional, Set
from tree_sitter import Node
from ...constants import KernelFlags


if TYPE_CHECKING:
    from . import (
        KernelInfoLike,
    )


class _TsExpressionsMixin:

    def _visit_expression(self, node:Node, scopes, ctx, is_write:bool)->None:
        node_type:str = node.type

        if node_type in ("identifier", "field_identifier"):
            root, _ = self._resolve(self._node_text(node), scopes)
            if root is not None:
                (ctx["writes"] if is_write else ctx["reads"]).add(root)
            return

        if node_type == "type_descriptor":
            # A cast's type sits here, and a cast is how C spells some *parenthesised
            # arithmetic*: `(w1 * ((float)(a[0]))) + (w2 * ...)` is parsed as a cast whose
            # type is `w1 * ((float)(a[0]))`, because `( expr ) + expr` and `( type ) expr`
            # are the same shape. The grammar therefore labels the operands
            # `type_identifier` / `abstract_*_declarator` even though they name variables.
            # Each of them is resolved as a plain name here: a real type name never resolves
            # to an argument, so a genuine cast contributes nothing while the ambiguous one
            # recovers exactly the reads and writes a reader would find.
            self._visit_type_descriptor(node, scopes, ctx)
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
            operator:str = (self._node_text(node)[0] if node.text else "*")
            operand:Optional[Node] = (node.named_children[0] if node.named_children else None)
            if operand is None:
                return
            if operator == "&":
                root, _ = self._resolve_expr_root(operand, scopes)
                if root is not None:
                    ctx["escapes"].add(root)
                else:
                    self._visit_expression(operand, scopes, ctx, False)
                    # &arr[i] takes an element address through a subscript: the bare array name does not resolve,
                    # so recursively collect the underlying array roots and mark them escaped (address exposed = may be written, e.g. atomicAdd(&arr[i]))
                    found:Set[str] = set()
                    self._collect_root_refs(operand, scopes, found)
                    ctx["escapes"].update(found)
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
                    member_bucket.setdefault(root, set()).add(self._node_text(field))

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
                    operator = self._node_text(child)
                    break

            lroot, lisptr = self._resolve_expr_root(left, scopes)
            if lisptr and left.type == "identifier":
                source:Optional[str] = self._alias_source(right, scopes)
                if source is not None:
                    self._bind_symbol(self._node_text(left), (source, True), scopes)
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


    def _visit_type_descriptor(self, node:Node, scopes, ctx)->None:
        """Count the names inside a cast's type as reads.

        The grammar spells both `(float)(a[0])` and the ambiguous `(w1 * (a[0])) + ...` as a
        `cast_expression` whose `type_descriptor` holds the left operand. A real type
        contributes only keywords (`primitive_type`, `type_identifier` naming a struct), and
        none of those resolves to a kernel argument, so resolving each name here costs one
        dictionary lookup when the cast is genuine -- and it is what keeps `w1` and `a[0]`
        alive when the cast is not.
        """
        if node.type in ("identifier", "field_identifier", "type_identifier"):
            root, _ = self._resolve(self._node_text(node), scopes)
            if root is not None:
                ctx["reads"].add(root)
            return
        for child in node.named_children:
            self._visit_type_descriptor(child, scopes, ctx)


    def _visit_call(self, node:Node, scopes, ctx)->None:
        children:List[Node] = node.named_children
        func:Node = children[0]
        func_name:str = (self._node_text(func) if func.type == "identifier" else "")
        # The argument list is found **by type**, not by position: a nested launch spells
        # `child<<<grid, block>>>(args)`, whose children are the callee, a
        # `kernel_call_syntax` node and then the argument list. Taking `children[1]` read
        # the launch geometry as the arguments, so the list was empty and every pointer
        # handed to the child looked untouched -- its flags came out as neither read nor
        # written, and skipping both copies broke the nested launch.
        arg_list:Node = next((child for child in children[1:]
                              if child.type == "argument_list"), None)
        actuals:List[Node] = (list(arg_list.named_children) if arg_list is not None else [])

        helper_flags:Optional[Dict[str, KernelFlags]] = None
        helper_info:Optional[KernelInfoLike] = None
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
                    # passing the pointer itself: not a data access, only propagate the helper's read/write sets
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
                else:
                    # &arr[i] passed to an unknown function (address exposed), e.g. atomicAdd(&arr[i], 1)
                    addr_taken:Set[str] = set()
                    self._collect_addr_taken_roots(actual, scopes, addr_taken)
                    ctx["escapes"].update(addr_taken)
