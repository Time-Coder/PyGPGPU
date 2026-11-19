from __future__ import annotations

import ast
from typing import Dict, Optional

from .predicates import (
    is_ptr,
)


class _OuterStoreMixin:
    """The outer-variable side of a plain-name assignment: what a store to a
    `nonlocal`/`global`-declared name means on the device, what is refused, and the
    rebind test a structure-pointer binding consults."""

    def _named_outer_target(self, target:ast.Name, node:ast.AST)->Optional[Dict[str, object]]:
        """Resolve a plain-name assignment to an outer variable, refusing what cannot work.

        A `nonlocal`/`global`-declared variable is written through its pointer, so
        `total = total + x` writes the scope value and the host copies it back after the
        launch. An undeclared outer name is refused: python would create a local there,
        which would silently shadow the scope variable instead of touching it.

        **A plain store from inside the work-item body is a race, so it is refused.** The
        variable is one location for the whole launch, so every work item overwrites it and
        the value that survives is whichever finished last -- the same hazard the *carried*
        scalars are checked for (see `_non_atomic_writes`). `total += x[i]` is the spelling
        that works, because it becomes a device atomic; a store guarded by something every
        work item agrees on (`if i == 0`) is a template for one work item's slice, and is
        also fine. What is not fine is an unguarded `counter = counter + x[i]`, which
        compiles and then answers one work item's partial value.
        """
        outer:Optional[Dict[str, object]] = self.outer_vars.get(target.id)
        if outer is None:
            return None
        if not outer["writable"]:
            self._error(
                f"cannot assign to the outer variable '{target.id}': without a "
                f"'nonlocal {target.id}' or 'global {target.id}' declaration this would "
                f"create a local variable, shadowing the outer one", node)
        if self.in_work_item_body and not self._store_is_guarded(node):
            self._error(
                f"'{target.id}' is one location shared by every work item (it is an outer "
                f"variable of the enclosing scope), so a plain store here races: whichever "
                f"work item finishes last decides the value. This is the same rule an "
                f"accumulated local follows -- write `{target.id} += value` (which becomes a "
                f"device atomic), or guard the store with the iteration index "
                f"(`if i == 0: {target.id} = ...`), or keep the work item's own value in a "
                f"local and reduce it afterwards", node)
        if is_ptr(outer["type"]):
            self._error(
                f"cannot rebind the outer array '{target.id}': the array object itself is "
                f"shared with the caller, so a device kernel can only write its elements", node)
        return outer

    def _name_is_reassigned(self, name:str)->bool:
        """Whether the body assigns `name` again after this binding.

        The tree is the translator's own deep copy, walked once per query and cached on
        the instance -- a name is asked once per binding site, and the answer decides
        whether the first binding of a structure pointer is an alias (the caller's
        object, read and written through) or a snapshot (python's own rebind rule: the
        later assignment rebinds the local and never reaches the caller).
        """
        cached:object = getattr(self, "_reassigned_names", None)
        if cached is None:
            cached = {}
            self._reassigned_names = cached
        if name in cached:
            return bool(cached[name])
        count:int = 0
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    # A tuple target (`ray, depth = new_ray, depth - 1`, the
                    # converted recursion's simultaneous rebind) counts every name
                    # it spells: the rebinding lands on each of them.
                    for item in ([target] if isinstance(target, ast.Name)
                                 else ast.walk(target) if isinstance(target,
                                                                     (ast.Tuple, ast.List))
                                 else []):
                        if isinstance(item, ast.Name) and item.id == name:
                            count += 1
            elif isinstance(node, ast.AnnAssign) and node.value is not None \
                    and isinstance(node.target, ast.Name) and node.target.id == name:
                count += 1
            elif isinstance(node, ast.AugAssign) \
                    and isinstance(node.target, ast.Name) and node.target.id == name:
                count += 1
        cached[name] = count > 1
        return count > 1
