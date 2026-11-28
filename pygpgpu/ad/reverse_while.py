# The reverse of a `while` loop: the replay scheme.
#
# Split from the statement walk by subject: a `while` is the one construct whose reverse
# cannot be a straight-line mirror of the forward, because its trip count is data. Each
# reverse step restores the carried values to the state before one iteration (by replaying
# from the saved init copies), runs that iteration's forward once more with the partials'
# saves, and then walks its adjoints -- quadratic in the trip count and still needing no
# memory, which is the right trade for the shapes kernels usually run.
from __future__ import annotations

import ast
from typing import List, Set, Tuple

from .ast_support import (
    assign,
    augassign,
    constant,
    copied,
    name,
    placed,
    store_name,
)
from .body_scan import _assign_targets


class _WhileReplayMixin:
    """The `while` half of the reverse walk, composed into ``GradBuilder``."""

    def walk_while(self, stmt:ast.While,
                   destroyed:Set[str])->Tuple[List[ast.stmt], List[ast.stmt]]:
        """The replay scheme: count the iterations, then walk them backwards without memory.

        The reverse walk replays the body once per iteration -- quadratic in the trip
        count and still needing no memory, which is the right trade for the shapes
        kernels usually run. Each reverse step restores the carried values to the state
        before one iteration (by replaying from the saved init copies), runs that
        iteration's forward once more with the partials' saves, and then walks its
        adjoints.
        """
        for sub in ast.walk(stmt):
            if isinstance(sub, (ast.Break, ast.Continue)):
                raise self._error(
                    "`break`/`continue` inside a `while` cannot be reversed: the trip "
                    "count the reverse walk replays would be wrong", sub)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                    and sub.func.id.startswith("atomic_"):
                raise self._error(
                    "an atomic update inside a `while` cannot be reversed: the replay "
                    "would apply it once per replay, not once per iteration", sub)
        carried:Set[str] = _carried_names(stmt)
        shared:Set[str] = carried & set(self.scan.carried_scalars)
        if shared:
            raise self._error(
                f"this `while` carries {sorted(shared)}, which the launch shares as one "
                f"location: the reverse walk replays the body once per iteration, and "
                f"the restore would race every other work item's write -- accumulate "
                f"into a per-work-item local inside the loop and add it to the carried "
                f"scalar once after it (`total += acc`)", stmt)
        counter:str = self.allocator.fresh("_ad_k")
        steps:str = self.allocator.fresh("_ad_r")
        replay:str = self.allocator.fresh("_ad_t")
        inits:List[ast.stmt] = []
        restore:List[ast.stmt] = []
        for value_name in sorted(carried):
            init_slot:str = self.allocator.fresh(f"_ad_init_{value_name}")
            inits.append(placed(assign(store_name(init_slot), name(value_name)),
                                stmt.lineno))
            restore.append(placed(assign(store_name(value_name), name(init_slot)),
                                  stmt.lineno))
        forward_body, reverse_body = self.walk_block(list(stmt.body), destroyed)
        loop:ast.While = copied(stmt)
        loop.body = forward_body + [
            placed(augassign(store_name(counter), ast.Add(), constant(1)), stmt.lineno)]
        counter_init:ast.stmt = placed(assign(store_name(counter), constant(0)),
                                       stmt.lineno)
        replay_loop:ast.While = copied(stmt)
        # The replay runs EXACTLY the steps the reverse step is at: its condition is the
        # counter, not the original one -- the original would replay until the forward's
        # own exit condition, which is the wrong trip count for step `_r`.
        replay_loop.test = name(replay) < name(steps)
        replay_loop.body = [copied(item) for item in forward_body] + [
            placed(augassign(store_name(replay), ast.Add(), constant(1)), stmt.lineno)]
        reverse_outer:ast.While = ast.While(
            test=name(steps) > constant(0),
            body=[placed(augassign(store_name(steps), ast.Sub(), constant(1)), stmt.lineno)]
                 + restore
                 + [placed(assign(store_name(replay), constant(0)), stmt.lineno),
                    replay_loop]
                 + [copied(item) for item in forward_body]
                 + reverse_body,
            orelse=[])
        reverse_outer.lineno = stmt.lineno
        reverse_outer.col_offset = 0
        # The step count is read AFTER the forward loop has counted it: assigning it
        # before the loop would snapshot the counter at zero and the reverse walk would
        # replay nothing.
        forward:List[ast.stmt] = inits + [counter_init, loop,
            placed(assign(store_name(steps), name(counter)), stmt.lineno)]
        return forward, [reverse_outer]



def _carried_names(stmt:ast.While)->Set[str]:
    """The names the while assigns, whose pre-state the reverse must be able to restore."""
    carried:Set[str] = set()
    for sub in ast.walk(stmt):
        if isinstance(sub, (ast.Assign, ast.AugAssign)):
            for target in _assign_targets(sub):
                if isinstance(target, ast.Name):
                    carried.add(target.id)
    return carried


