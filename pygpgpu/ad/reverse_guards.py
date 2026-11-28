# The scan-level refusals of the fused reverse walk: what cannot be differentiated,
# said out loud before any statement is walked.
#
# Split from the statement walk by subject, per the mixin extraction the file-size policy
# names: `reverse.py` orders the walk; this module is the gate in front of it -- the
# multi-pass guard, the write analysis (tracked arrays, members, the loss and the output
# parameter), the atomic gate and the host-region refusal.
from __future__ import annotations

import ast
from typing import Dict, List, Set

from .body_scan import walk_skipping_shape


class _ScanGuardsMixin:
    """The pre-walk refusals, composed into ``GradBuilder``."""

    def _check_scanned(self)->None:
        """The scan-level refusals: what this form cannot differentiate, said out loud."""
        if len(self.scan.dispatches) > 1:
            # A body with several magic loops launches as several kernels (the phase
            # machinery); the fused adjoint below would see only the first pass and the
            # later passes' contributions would silently vanish from every gradient.
            raise self._error(
                "this body has several magic loops, so a launch runs it as several "
                "kernels; the fused adjoint is single-pass and would silently drop the "
                "later passes' gradients -- split the passes into separate @kernel "
                "functions and chain them with pygpgpu.Tape", self.scan.dispatches[1])
        stored:Dict[str, List[ast.stmt]] = self.scan.store_targets()
        # A tracked array is never written: the adjoint needs the value the forward read,
        # which any store overwrites (the plan's in-place refusal). The loss and the
        # output parameter are the only legal writes -- and every write must land in a
        # parameter, never in a receiver or object member, whose gradient slot the fused
        # form does not carry.
        allowed_writes:Set[str] = {self.loss_name or ""}
        if self.cot_of_out is not None:
            allowed_writes.add(self.cot_of_out.out_param)
        for param, stores in stored.items():
            if param not in allowed_writes:
                if self.scan.is_member_root(param):
                    raise self._error(
                        f"`{param}` is a receiver or object member, and a member store "
                        f"has no gradient slot in the fused form: pass the member as its "
                        f"own array argument and store into that", stores[0])
                if param in self.wrt:
                    raise self._error(
                        f"`{param}` is tracked (`requires_grad`): a differentiable body "
                        f"reads it but never writes it in place -- the adjoint needs the "
                        f"value the forward read, which this store would overwrite",
                        stores[0])
                raise self._error(
                    f"a differentiable kernel writes only its output parameter and the "
                    f"declared loss; `{param}` is neither -- spell `out=` or `loss=` for "
                    f"the parameter this body answers through", stores[0])
        reads:Dict[str, int] = self.scan.value_reads()
        if self.loss_name is not None and self.loss_name in reads:
            raise self._error(
                f"`{self.loss_name}` is the loss and the body reads it: a value the "
                f"launch is still accumulating cannot be read mid-flight -- accumulate "
                f"into it and let the caller read it after the launch",
                self.scan.store_targets().get(self.loss_name, [None])[0]
                or next(node for node in ast.walk(self.scan.tree)
                        if isinstance(node, ast.Name) and node.id == self.loss_name))
        for written in stored:
            if written in reads:
                raise self._error(
                    f"`{written}` is both written and read: the adjoint needs the "
                    f"pre-state the store overwrote -- accumulate into a local and "
                    f"store once", stored[written][0])
        for call_node in self.scan.atomic_calls():
            spell:str = call_node.func.id
            if spell == "atomic_add" and self.loss_name is not None \
                    and isinstance(call_node.args[0], ast.Name) \
                    and call_node.args[0].id == self.loss_name:
                continue
            raise self._error(
                f"`{spell}` is not differentiable here: the only supported atomic is "
                f"`atomic_add` into the `loss=` parameter, whose adjoint scatters the "
                f"cotangent the loss slot carried", call_node)
        for stmt in ast.walk(self.scan.tree):
            if isinstance(stmt, (ast.Raise, ast.Assert)):
                raise self._error(
                    f"`{type(stmt).__name__}` is not differentiable: the adjoint of this "
                    f"statement has no device meaning", stmt)

    def _check_host_region(self, prefix:List[ast.stmt])->None:
        """The host region runs python, and this package does not do numpy AD."""
        for stmt in prefix:
            for sub in walk_skipping_shape(stmt):
                if isinstance(sub, ast.Name) and sub.id in self.wrt \
                        and isinstance(sub.ctx, ast.Load):
                    raise self._error(
                        f"`{sub.id}` is tracked and this statement runs on the host, "
                        f"above the work-item dispatch: the host has no AD in v1 -- write "
                        f"a loss kernel, or move the read into the loop", sub)

