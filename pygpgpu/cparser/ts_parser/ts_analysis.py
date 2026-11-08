# The analysis of what a declaration means: which parameters are pointers,
# which arguments are passed by address, and what a kernel's body does.
from __future__ import annotations
from typing import TYPE_CHECKING
import copy
from typing import Dict, List, Optional, Set, Tuple
from tree_sitter import Node
from ...constants import KernelFlags


if TYPE_CHECKING:
    from . import (
        AnalysisContext,
        VarInfoLike,
        KernelInfoLike,
    )


class _TsAnalysisMixin:

    def _analyze_kernels(self)->None:
        for name, kernel_info in self._kernel_infos.items():
            body:Optional[Node] = self._function_body(self._kernel_bodies.get(name))
            if body is None:
                continue
            self._analyze_kernel(kernel_info, body)


    def _analyze_kernel(self, kernel_info:KernelInfoLike, body:Node)->None:
        reads, writes, escapes, member_reads, member_writes = self._analyze_body(kernel_info, body)
        for name, flags in self._param_flags(kernel_info, reads, writes, escapes).items():
            kernel_info.args[name].kernel_flags = flags
        self._apply_member_flags(kernel_info, escapes, member_reads, member_writes)


    def _pointer_members(self, arg:VarInfoLike)->Dict[str, VarInfoLike]:
        """Pointer members of a struct-typed argument (empty for non-struct args)."""
        return {}


    def _apply_member_flags(self, kernel_info:KernelInfoLike, escapes:Set[str], member_reads:Dict[str, Set[str]], member_writes:Dict[str, Set[str]])->None:
        """Derives per-member KernelFlags for pointer members of by-value struct args.

        Flags are stored per-arg (arg.member_infos) so each kernel keeps its own
        copy policy; members never touched fall back to Readed|Writed.
        """
        for name, arg in kernel_info.args.items():
            members:Dict[str, VarInfoLike] = self._pointer_members(arg)
            if not members:
                continue

            member_infos:Dict[str, VarInfoLike] = {}
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

                member_copy:VarInfoLike = copy.copy(member)
                member_copy.kernel_flags = flags
                member_infos[member_name] = member_copy

            arg.member_infos = member_infos


    def _call_flags(self, func_name:str)->Optional[Dict[str, KernelFlags]]:
        """Per-parameter KernelFlags of a helper function, analyzed on demand."""
        if func_name in self._helper_flags:
            return self._helper_flags[func_name]

        func_info:Optional[KernelInfoLike] = self._function_infos.get(func_name)
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


    def _param_flags(self, info:KernelInfoLike, reads:Set[str], writes:Set[str], escapes:Set[str])->Dict[str, KernelFlags]:
        """The copy policy of every pointer argument: what is uploaded, what is read back.

        A pointer the body never touches is copied **neither way**. The analysis knows the
        body, so a buffer that appears nowhere in it cannot contribute or receive anything,
        and moving it costs a copy for no result -- which is what a kernel that reserves an
        argument for a code path it does not take would otherwise pay on every launch. That
        is why the "no flags" case is left empty rather than widened to both.

        A buffer the body **writes** is uploaded as well, because the read-back copies the
        *whole* buffer into the caller's array: a buffer that was never uploaded would
        replace every element the kernel did not write with the contents of a fresh device
        buffer (zeros). Python keeps those elements, so `out[i, j, 0] = ...` on a 3-channel
        image must leave channels 1 and 2 as the caller had them.

        An explicit `write_only` qualifier no longer changes the copy policy. It described
        what the *shader* may assume, and this host cannot honour the saving it implied for
        the same reason as above: the read-back is whole-buffer, so an upload that was
        skipped would come back as zeros in every element the kernel left alone. The
        qualifier is therefore a statement about the body's uses, already captured by the
        analysis, not an instruction to stop copying.
        """
        result:Dict[str, KernelFlags] = {}
        for name, arg in info.args.items():
            if not self._is_mem_arg(arg):
                result[name] = KernelFlags.Readed
                continue

            if name in escapes:
                # The pointer escaped (handed to a helper or stored): its uses are unknown,
                # so it keeps both directions.
                flags:KernelFlags = (KernelFlags.Readed | KernelFlags.Writed)
            else:
                flags = ((KernelFlags.Readed if name in reads else KernelFlags(0)) |
                         (KernelFlags.Writed if name in writes else KernelFlags(0)))

            # A `const`/read-only qualifier is authoritative: the body cannot write through
            # it, so there is nothing to read back. A `write_only` one is not consulted --
            # see the docstring.
            if arg.readonly:
                flags = KernelFlags.Readed
            elif flags & KernelFlags.Writed:
                flags = flags | KernelFlags.Readed

            result[name] = flags
        return result


    def _function_body(self, node:Optional[Node])->Optional[Node]:
        if node is None or node.type != "function_definition":
            return None
        return next((c for c in node.named_children if c.type == "compound_statement"), None)


    def _analyze_body(self, info:KernelInfoLike, body:Node)->Tuple[Set[str], Set[str], Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
        ctx:AnalysisContext = {
            "reads": set(), "writes": set(), "escapes": set(),
            "member_reads": {}, "member_writes": {}, "struct_args": set()
        }
        roots:Dict[str, Tuple[Optional[str], bool]] = {}
        for name, arg in info.args.items():
            roots[name] = (name, self._is_mem_arg(arg))
        ctx["struct_args"] = {name for name, arg in info.args.items() if self._pointer_members(arg)}

        scopes:List[Dict[str, Tuple[Optional[str], bool]]] = [roots]
        for child in body.named_children:
            self._visit_statement(child, scopes, ctx)

        return ctx["reads"], ctx["writes"], ctx["escapes"], ctx["member_reads"], ctx["member_writes"]
