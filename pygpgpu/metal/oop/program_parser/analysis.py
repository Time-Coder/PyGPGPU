"""What a Metal entry point's body does with each of its pointers.

This is the piece the declaration scanner never had, and the reason the Metal
backend paid for a read-back it did not need: every pointer was copied both ways.
The walk below answers which arguments the body reads and which it writes, so the
launch can skip a read-back (a read-only buffer) or both copies (an argument the
body never mentions) -- the same policy the C backends get from their tree-sitter
analysis.

Four rules are load-bearing and match the shared analysis:

* **A written buffer is still uploaded.** The read-back copies the *whole* buffer,
  so skipping the upload would replace every element the body left alone with the
  contents of a fresh device buffer.
* **A pointer that escapes keeps both directions.** Handing a pointer to a device
  function (or taking its address) leaves its uses unknowable here; guessing
  read-only would silently drop a write.
* **A `const` qualifier is authoritative.** The body cannot store through it, so
  there is nothing to read back, whatever the walk believes it saw.
* **An unrecognized node is descended into, never skipped**, so a construct a later
  grammar version grows still contributes its reads and writes.

The distinction that makes the walk correct is **which positions are lvalues**. A
`subscript_expression` in a store's destination is a write; everywhere else it is a
read. Only the lvalue path (`subscript_expression`, `field_expression`,
`parenthesized_expression`, `pointer_expression`, `cast_expression`) passes
"this is a write" down to its base; every other node's children are reads.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set, Tuple

from tree_sitter import Node

from ....constants import KernelFlags

if TYPE_CHECKING:
    from ..program_info import KernelInfo, VarInfo


# The nodes that form an lvalue path: a write's destination may be wrapped in any of
# these, and the "this is a write" flag travels through them to the base name.
_LVALUE_PATH: frozenset = frozenset((
    "subscript_expression", "field_expression", "parenthesized_expression",
    "pointer_expression", "cast_expression",
))

# The declarator kinds that can introduce a local pointer alias (`device float* p = a`).
_POINTER_DECLARATORS: frozenset = frozenset((
    "pointer_declarator", "array_declarator", "reference_declarator",
))


class _MetalAnalysisMixin:
    """The body-analysis half of the Metal walk."""

    def _analyze_kernels(self) -> None:
        for name, kernel_info in self._kernel_infos.items():
            body: Node = self._kernel_bodies.get(name)
            if body is None:
                continue
            self._analyze_kernel(kernel_info, body)

    def _analyze_kernel(self, kernel_info: "KernelInfo", body: Node) -> None:
        reads, writes = self._analyze_body(kernel_info, body)
        for arg in kernel_info.args.values():
            arg.kernel_flags = self._copy_policy(arg, reads, writes)

    def _copy_policy(self, arg: "VarInfo", reads: Set[str],
                     writes: Set[str]) -> KernelFlags:
        """The copy policy of one argument, from the body and the declaration together."""
        if not arg.is_ptr:
            return KernelFlags.Readed
        if arg.readonly:
            # The qualifier is authoritative: the body cannot store through it.
            return KernelFlags.Readed

        flags = KernelFlags(0)
        if arg.name in reads:
            flags |= KernelFlags.Readed
        if arg.name in writes:
            # A write needs a read-back, and a whole-buffer read-back needs the upload.
            flags |= KernelFlags.Readed | KernelFlags.Writed
        return flags

    # ---- the walk ----

    def _analyze_body(self, kernel_info: "KernelInfo",
                      body: Node) -> Tuple[Set[str], Set[str]]:
        """Walk one body and answer which arguments are read and which are written."""
        roots: Dict[str, str] = self._pointer_roots(kernel_info, body)
        reads: Set[str] = set()
        writes: Set[str] = set()

        def root_of(name: str) -> str:
            return roots.get(name, "")

        def use(name: str, is_write: bool) -> None:
            """Record one use of a name, resolved to the argument it aliases."""
            root: str = root_of(name)
            if not root:
                return
            if is_write:
                writes.add(root)
            else:
                reads.add(root)

        def escape(node: Node) -> None:
            """Keep both directions for a pointer that leaves this body's sight."""
            root: str = self._root_name(node, roots)
            if root:
                writes.add(root)
                reads.add(root)

        def visit(node: Node, is_write: bool = False) -> None:
            kind: str = node.type

            if kind == "assignment_expression":
                children: List[Node] = node.named_children
                if len(children) >= 2:
                    # `+=` reads the old value too, so the destination is both.
                    compound: bool = self._text(node).find("=") > 1
                    visit(children[0], True)
                    if compound:
                        visit(children[0], False)
                    visit(children[1], False)
                return

            if kind == "update_expression":
                # `x++` / `--x`: reads the old value and writes the new one.
                for child in node.named_children:
                    visit(child, True)
                    visit(child, False)
                return

            if kind == "subscript_expression":
                base = node.named_children[0] if node.named_children else None
                if base is not None and base.type == "identifier":
                    use(self._text(base), is_write)
                elif base is not None:
                    # `rows[i].field` reaches here as a field over a subscript: the
                    # base is itself an lvalue path and keeps the flag.
                    visit(base, is_write)
                # The index expressions are always reads.
                for child in node.named_children[1:]:
                    visit(child, False)
                return

            if kind == "field_expression":
                base = node.named_children[0] if node.named_children else None
                if base is not None:
                    visit(base, is_write)
                if is_write:
                    # A member store writes the record the member lives in.
                    root = self._root_name(node.named_children[0], roots) if base else ""
                    if root:
                        writes.add(root)
                return

            if kind == "pointer_expression":
                # `&x[i]` takes an address: the pointer escapes, so both directions.
                escape(node)
                for child in node.named_children:
                    visit(child, False)
                return

            if kind == "call_expression":
                # A pointer argument handed to a function is an escape -- the callee may
                # write through it. A scalar or element argument is an ordinary read.
                for child in node.named_children:
                    if child.type == "argument_list":
                        for argument in child.named_children:
                            if self._adorned_pointer(argument, roots):
                                escape(argument)
                            visit(argument, False)
                        continue
                    visit(child, False)
                return

            if kind == "identifier":
                use(self._text(node), is_write)
                return

            for child in node.named_children:
                visit(child, False if kind not in _LVALUE_PATH else is_write)

        visit(body, False)
        return reads, writes

    def _adorned_pointer(self, node: Node, roots: Dict[str, str]) -> bool:
        """Whether a call argument hands over a pointer rather than a value.

        `foo(a)` with `a` a pointer parameter, `foo(&a[i])`, `foo(parts)` -- all let the
        callee write memory this body can see. `foo(a[i])` passes a value and does not.
        """
        current: Node = node
        while current is not None:
            if current.type == "pointer_expression":
                return True
            named = current.named_children
            if current.type == "identifier":
                return bool(self._root_name(current, roots))
            if current.type in ("parenthesized_expression", "cast_expression"):
                current = named[0] if named else None
                continue
            return False
        return False

    # ---- local pointer aliases ----

    def _pointer_roots(self, kernel_info: "KernelInfo", body: Node) -> Dict[str, str]:
        """Map every local pointer name to the argument it ultimately names.

        `device float* p = a;` makes `p` another spelling of `a`, so a store through
        `p` writes the caller's array. Without this the analysis would leave `a`
        read-only and the host would drop the write -- silently.
        """
        roots: Dict[str, str] = {name: name for name, arg in kernel_info.args.items()
                                 if arg.is_ptr}
        changed: bool = True
        while changed:
            changed = False
            for declaration in self._declarations(body):
                alias: Tuple[str, str] = self._alias_of(declaration)
                if not alias[0] or not alias[1]:
                    continue
                root: str = roots.get(alias[1], "")
                if root and roots.get(alias[0]) != root:
                    roots[alias[0]] = root
                    changed = True
        return roots

    def _alias_of(self, declaration: Node) -> Tuple[str, str]:
        """The `(new_name, source_name)` pair one pointer declaration binds."""
        name: str = ""
        source: str = ""
        for child in declaration.named_children:
            if child.type in _POINTER_DECLARATORS:
                name, _is_ptr, _shape = self._walk_declarator(child)
                # `*p = a` and `*p = &a[0]`: the initializer sits inside the declarator.
                for inner in child.named_children:
                    if inner.type == "identifier" and not source:
                        source = self._text(inner)
            elif child.type == "init_declarator":
                for inner in child.named_children:
                    if inner.type in _POINTER_DECLARATORS:
                        name, _is_ptr, _shape = self._walk_declarator(inner)
                    elif inner.type == "identifier" and not source:
                        source = self._text(inner)
                    elif inner.type == "pointer_expression":
                        for deep in inner.named_children:
                            if deep.type == "subscript_expression" and deep.named_children:
                                source = self._text(deep.named_children[0])
        return name, source

    def _declarations(self, body: Node):
        """Every declaration node under one body."""
        stack: List[Node] = [body]
        while stack:
            node = stack.pop()
            if node.type == "declaration":
                yield node
            stack.extend(node.named_children)

    def _root_name(self, node: Node, roots: Dict[str, str]) -> str:
        """The argument a subscript/field/pointer expression ultimately names."""
        current: Node = node
        while current is not None:
            kind: str = current.type
            named: List[Node] = current.named_children
            if kind == "identifier":
                return roots.get(self._text(current), "")
            if kind in _LVALUE_PATH or kind in ("binary_expression", "unary_expression"):
                current = named[0] if named else None
                continue
            return ""
        return ""


__all__ = ["_MetalAnalysisMixin"]
