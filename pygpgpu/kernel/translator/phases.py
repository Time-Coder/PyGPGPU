from __future__ import annotations
import ast
from typing import (Dict, List, Optional, Set)
from .source_text import (
    _has_device_return,
    _returns_a_value,
)
from .notation import (
    DeviceType,
    PTR,
)
from .walk import (
    _assigned_name_of,
    _creation_target_of,
    _magic_loop_end,
    _magic_loop_index,
)


class _PhasesMixin:

    def translate(self)->None:
        body:List[ast.stmt] = self.tree.body
        # A local pointer (`src = a`) is known before the body is walked: the magic-loop
        # scan asks about `enumerate(src)` before that statement is reached (see
        # _collect_array_pointers).
        self._collect_array_pointers(body)
        # `weights = [1.0, 2.0]` feeding a loop is a compile-time table, recorded before the
        # walk so the binding emits no device code and the loop can read it.
        self._collect_constant_lists(body)
        # Outer-scope names and host-created objects must be known before the body is
        # visited: every use of them compiles against the parameter that will carry them,
        # and the magic-loop scan asks which arrays exist before any statement is reached
        # (a created array can be the loop's iteration domain).
        self._bind_outer_variables()
        if self.is_kernel:
            self._bind_host_values(body)
            # Whether any return site carries a value decides the device signature, and
            # that is known only after the whole body has been seen; every site's code
            # depends on it, so the body is scanned once first.
            self.has_returns = _returns_a_value(body)
            # Whether any of those returns is device code, which is a different question:
            # a tail return runs on the host and claims nothing (see render_return_params).
            self._claims_on_device = _has_device_return(body)
            # A local initialized above the loop and written inside it is one location for
            # the whole launch, not one per work item -- see _carried_scalars.
            self._bind_carried_scalars(body)
            for offender in self._non_atomic_writes(body, self.carried_scalars):
                self._error(
                    "this variable is one location shared by every work item (it is "
                    "accumulated inside the loop and read after it), so a plain store here "
                    "races: whichever work item finishes last decides the value. Use `+=` "
                    "(which becomes a device atomic), or keep the per-work-item value in a "
                    "different variable and reduce it afterwards", offender)
            # A body with several magic loops is several *passes* over the work items. Each
            # one is translated on its own into a kernel of its own, because a second loop
            # inside one kernel would be run by every work item instead of once. They are
            # launched back to back on the same stream, so the arrays stay on the device.
            self.phases = self.magic_loop_phases(body)
            if len(self.phases) == 1:
                self._translate_magic_loop(body)
            else:
                self._translate_phase_kernels()
        else:
            for stmt in body:
                self._visit_statement(stmt)
            if self.return_type is None:
                self.return_type = "void"


    def _translate_phase_kernels(self)->None:
        """Translate one kernel per magic loop, each owning its own statements.

        A phase is a separate kernel, so a name it reads has to be one of its own
        parameters. The names a *different* phase computes therefore have to cross the
        boundary, and how depends on what the name holds:

        * a **created array** (`mid = np.zeros(4)`) is built once on the host and travels as
          a trailing parameter, exactly as it does in a single-phase body -- whether it sits
          above the first loop or between two of them;
        * a **plain scalar** (`scale = 10`) is copied into the body of every later phase and
          recomputed there. It is a value each work item can derive for itself, so shipping
          it as a parameter would add an argument nothing needs, and it is what keeps a
          body's `k = 2 * factor` readable across the split.
        """
        # The statements every phase shares: those above the first loop, plus the gaps
        # between the loops (a creation there has to exist before the pass that reads it).
        header:List[ast.stmt] = self.phases[0][:_magic_loop_index(self.phases[0])]
        carried:List[ast.stmt] = []
        for index in range(len(self.phases) - 1):
            gap:List[ast.stmt] = self.phases[index][_magic_loop_end(self.phases[index]):]
            for stmt in gap:
                # Every gap statement runs on the host: python executes it once, between the
                # two passes. A creation is bound by _bind_host_values, which already runs on
                # the host; anything else is a value the next pass receives as a parameter.
                #
                # A statement the interpreter *already ran* (its line is in
                # host_statement_lines) is not copied into the header: copying it made
                # every phase emit device code for it under the same function name, and
                # the two definitions then collided. Its value travels as a host scalar
                # instead (see _bind_host_statement), which is what the next pass needs.
                if (int(getattr(stmt, "lineno", 0) or 0) in self.host_statement_lines
                        and _assigned_name_of(stmt) is not None):
                    carried.append(stmt)
                    continue
                header.append(stmt)
                if not self._is_device_creation(stmt) and _assigned_name_of(stmt) is not None:
                    carried.append(stmt)

        # The names a gap computed on the host: each is a parameter of the pass that reads
        # it, so it is declared (and marked like a carried slot) before the body is visited.
        gap_values:List[str] = [
            name for name in (_assigned_name_of(stmt) for stmt in carried)
            if name is not None]
        self.gap_names = list(gap_values)
        # The statements after each phase's loop, in phase order: the host runs region i
        # between launch i and launch i + 1 (see PyKernel._run_host_region).
        self.phase_gaps:List[List[ast.stmt]] = [
            list(self.phases[index][_magic_loop_end(self.phases[index]):])
            for index in range(len(self.phases) - 1)]

        for index, phase_body in enumerate(self.phases):
            # The gap that follows this phase belongs to the *next* one (it is either a
            # creation the host builds or a value the next pass recomputes), so it is cut
            # from this phase's body: emitted here it would be device code, and a creation
            # has no device form.
            body:List[ast.stmt] = (list(phase_body[:_magic_loop_index(phase_body) + 1]
                                        if index < len(self.phases) - 1 else phase_body))

            # Through the facade, lazily: this mixin is imported by the module that
            # defines the class, and a top-level import would be a cycle.
            from . import FunctionTranslator
            phase:FunctionTranslator = FunctionTranslator(
                self.func, self.param_types, self.backend, self.resolve_global,
                self.device_cache, is_kernel=True, self_param=self.self_param,
                self_attr_types=self.self_attr_types, self_owner=self.self_owner,
                launch_backend=self.launch_backend,
                host_creation_values=self.host_creation_values,
                host_executed_lines=sorted(self.host_statement_lines))
            # The tree is the same function, so the phase is visited by transplanting its
            # statements onto the phase translator's own tree.
            phase.tree.body = body
            # The gap this phase precedes is host code, but the phase still has to see the
            # names it feeds: a scalar the loop accumulates and the gap reads keeps its
            # shared slot in the writing phase (see _carried_scalars).
            phase.phase_gap = list(self.phases[index][_magic_loop_end(self.phases[index]):])
            phase._bind_outer_variables()
            phase.phase_carried_hint = list(self.carried_scalars)
            # A phase skips the main translate() walk, so the shared-slot binding has to run
            # here: without it, a scalar the loop accumulates compiles as per-item scratch
            # and the reduction silently loses every contribution but one work item's.
            phase._bind_carried_scalars(body)
            for offender in phase._non_atomic_writes(body, phase.carried_scalars):
                phase._error(
                    "this variable is one location shared by every work item (it is "
                    "accumulated inside the loop and read after it), so a plain store here "
                    "races: whichever work item finishes last decides the value. Use `+=` "
                    "(which becomes a device atomic), or keep the per-work-item value in a "
                    "different variable and reduce it afterwards", offender)
            # A later phase does not contain the shared creations, so it binds them here;
            # the object is the same host object for every phase.
            phase._bind_host_values(header + body if index else body)
            used:Set[str] = {node.id for node in ast.walk(ast.Module(body=body, type_ignores=[]))
                             if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                             and node.id in gap_values}
            # A gap value is one of the pass's own parameters, but it must be declared
            # *after* the body is visited: the shape extents are appended while translating,
            # and inserting a parameter ahead of them shifts every later argument.
            for name in gap_values:
                # A value the interpreter bound between the passes (a host-region statement)
                # travels as a host scalar of its *own* type (see _bind_host_statement):
                # declaring it here as a carried `int*` slot as well made the launch pass
                # the string it was computed from into an int parameter.
                if name in phase.host_scalars:
                    continue
                if name in used and name not in phase.scopes[-1]:
                    phase.scopes[-1][name] = ("carried", name)
            phase.has_returns = _returns_a_value(body)
            # Every phase is its own kernel, so whether one of them claims is a fact about
            # its own slice: a pass whose only return sits in its tail hands it back on the
            # host and needs no claim word (see render_return_params).
            phase._claims_on_device = _has_device_return(body)
            phase._translate_magic_loop(body)
            # The statements after this phase's own loop. For the last phase that is the
            # body's tail, which the host runs after the final launch; for an earlier one it
            # is the gap that feeds the next pass, and the launch already runs it as a region
            # (see _run_host_region). Recording it on the phase keeps the tail honest for a
            # multi-loop body: only the main translator used to keep one, so a body split
            # into passes lost its tail entirely and a `return out` after the last loop
            # answered None.
            phase.body_tail = list(body[_magic_loop_end(body):])
            for name in gap_values:
                if name in phase.host_scalars:
                    continue
                if name in used and name not in dict(phase.params):
                    phase.params.append((name, (PTR, "int")))
                    phase.carried_scalars.append(name)
            self.phase_translators.append(phase)
            # The host builds one argument list for the whole call, so every phase's
            # parameters have to be known here: a value only a later phase needs (an array
            # created between the loops) is still a parameter of this translation.
            for name, value_type in phase.params:
                if name not in dict(self.params):
                    self.params.append((name, value_type))
            # The same for outer-scope values: the host builds one launch, so every name any
            # phase reads has to be known here.
            for name, outer in phase.outer_vars.items():
                if name not in self.outer_vars:
                    self.outer_vars[name] = outer


    def _is_device_creation(self, stmt:ast.stmt)->bool:
        """True when ``stmt`` builds data, as opposed to computing a value."""
        value:Optional[ast.expr] = _creation_target_of(stmt)
        return value is not None and self._is_creation_expression(value)


    def magic_loop_phases(self, body:List[ast.stmt])->List[List[ast.stmt]]:
        """Split a kernel body into one statement list per magic loop.

        A body with a single magic loop is one phase and the split is trivial. Two or more
        are separate *passes* over the work items, and they are translated as consecutive
        launches rather than one kernel: a second loop inside one kernel would be executed
        by every work item instead of once (see the refusal in _visit_statement).

        The statements before the first loop stay with it, since they belong to the first
        pass, and everything from the last loop to the end of the body is the last phase.
        Each returned list has exactly one magic loop, which is what the single-phase
        translation below expects.
        """
        # Only loops *shaped* like a magic loop count, and only from the second one on: the
        # first is the work-item dispatch, so the phases start at it and each later one adds
        # a boundary.
        loops:List[int] = [
            index for index, stmt in enumerate(body)
            if isinstance(stmt, ast.For) and self._is_secondary_magic_loop(stmt)]
        if len(loops) < 2:
            return [body]
        boundaries:List[int] = loops[1:]
        phases:List[List[ast.stmt]] = []
        start:int = 0
        for boundary in boundaries:
            phases.append(body[start:boundary])
            start = boundary
        phases.append(body[start:])
        return phases


    def _is_secondary_magic_loop(self, node:ast.For)->bool:
        """True for a `for` shaped like the work-item dispatch, at the top level.

        The shape is the one the first loop has -- `range(arr.shape[0])`, `enumerate(arr)`,
        `np.ndindex(...)` -- but its position is what matters: only the first one is the
        launch. A `range` over something that is not an array's shape (`range(8)`) does not
        name a work-item domain, so it is an ordinary serial loop and stays allowed, exactly
        as it was before the several-pass split existed.

        An argument that is a local alias of an array (`src = a`, then `enumerate(src)`) is
        the same domain: resolving the alias here is what keeps a body with one dispatch
        loop from reading as a body with two.
        """
        iterator:ast.expr = node.iter
        if isinstance(iterator, ast.Call):
            func:ast.expr = iterator.func
            if isinstance(func, ast.Name) and func.id == "enumerate":
                return True
            if isinstance(func, ast.Name) and func.id == "range":
                # Only a data-shaped range is a dispatch; a literal count is a plain loop.
                return self._shape0_range_param(iterator) is not None
            if (isinstance(func, ast.Attribute) and func.attr in ("ndindex", "ndenumerate")
                    and isinstance(func.value, ast.Name)):
                return True
        return False


    def _visit_region(self, statements:List[ast.stmt], after_magic_loop:bool)->None:
        """Visit a run of statements, marking whether they run once per work item.

        Everything after the magic loop is emitted into every work item, so a value
        `return` there hands back one work item's own local -- not the single value the
        python function would compute after its loop. `_visit_kernel_return` refuses such
        a return; an array return stays allowed, since the object is the same whichever
        work item names it.
        """
        previous:bool = self.after_magic_loop
        self.after_magic_loop = previous or after_magic_loop
        try:
            for stmt in statements:
                self._visit_statement(stmt)
        finally:
            self.after_magic_loop = previous


    def _bind_carried_scalars(self, body:List[ast.stmt])->None:
        """Give every `carried` local a one-element buffer parameter instead of a device `T`.

        The parameter is appended in the same trailing position an outer value uses, and it
        is writable, so the body reads and writes it through the pointer and the host reads
        the final value back after the launch.
        """
        names:List[str] = list(self._carried_scalars(body))
        # Names the whole body carried that this phase initializes or writes: the read that
        # made them shared lives in a later pass, which this phase's slice no longer shows.
        hint:Set[str] = set(getattr(self, "phase_carried_hint", []) or [])
        magic_index:int = _magic_loop_index(body)
        written:Set[str] = set()
        for stmt in body[magic_index if magic_index < len(body) else len(body):]:
            for node in ast.walk(stmt):
                if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    written.add(node.target.id)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            written.add(target.id)
        for name in sorted(hint):
            if name not in names and name in written:
                names.append(name)
        for name in names:
            declared:DeviceType = self._lookup(name)
            if declared is not None:
                continue
            # The slot is the carried value's own width: the initializer above the loop
            # (`total = 0.0` is a float, `total = 0` an int) says what python answers,
            # and the atomic below is typed the same way. The slot used to be `int` for
            # every carrier, which made a float sum's `atomicAdd` an int atomic -- the
            # contributions were truncated (`0.5 + 0.25` answered 0), and the
            # interpreter, which runs python's own arithmetic, answered 0.75.
            element_t:object = "int"
            init_stmt:Optional[ast.stmt] = next(
                (stmt for stmt in body[:_magic_loop_index(body)]
                 if _assigned_name_of(stmt) == name), None)
            if init_stmt is not None:
                init_value:Optional[ast.expr] = _creation_target_of(init_stmt)
                if isinstance(init_value, ast.Constant):
                    if isinstance(init_value.value, float):
                        element_t = self.type_map.get("float", "float")
                    elif isinstance(init_value.value, int) \
                            and not isinstance(init_value.value, bool):
                        element_t = self.type_map.get("int", "int")
            self.carried_scalars.append(name)
            self.params.append((name, (PTR, element_t)))
            self._define(name, ("carried", name), body[0])


    def _carried_scalars(self, body:List[ast.stmt])->List[str]:
        """The locals assigned *above* the magic loop and written *inside* it.

        Such a name is a device `T` today, so every work item accumulates into its own copy
        and the result is one work item's partial sum rather than the whole one -- silently,
        because nothing about the shape of the program looks wrong. Python has one `total`:
        it is initialized once, then updated by every iteration in turn.

        Carrying it as a one-element array is the same shape the framework already uses for a
        writable outer scalar (`T*`, read and written through the pointer), so the device
        shares one location across the whole launch and the host can read the final value.
        The statements above the loop still run on the host -- that is where the initial
        `total = 0` comes from -- and the kernel receives the slot, not the value.
        """
        magic_index:int = _magic_loop_index(body)
        initialized:Dict[str, ast.stmt] = {}
        for stmt in body[:magic_index]:
            name:Optional[str] = _assigned_name_of(stmt)
            if name is None:
                continue
            initialized[name] = stmt
        written:set = set()
        for stmt in body[magic_index:]:
            for node in ast.walk(stmt):
                if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    written.add(node.target.id)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            written.add(target.id)
        # Only a name the loop's *result* carries needs the shared location: one whose value
        # is still read after the loop. A name used only inside the loop is each work item's
        # own scratch, and making it shared would turn a per-item value into a race -- the
        # behaviour is deterministic today and would stop being so.
        # Everything after the loop *itself*: the loop's body runs once per work item, so a
        # name read there is per-item scratch, while one read after the loop is the value the
        # launch is supposed to have produced.
        after:List[ast.stmt] = body[magic_index + 1:]
        used_after:set = {
            node.id for stmt in after for node in ast.walk(stmt)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        carried:List[str] = []
        for name in initialized:
            if name in written and name in used_after:
                carried.append(name)
        return carried


    def _non_atomic_writes(self, body:List[ast.stmt], names:List[str]) -> List[ast.stmt]:
        """The statements inside the loop that *store* one of ``names`` without `+=`.

        A carried slot is one location for the whole launch, so a plain store there is a
        race: every work item overwrites it, and the value that survives is whichever
        finished last. `total += x[i]` is fine (it becomes an atomic), and so is a
        conditional update built on it; `best = x[i]` inside an `if` is not, and it has to be
        refused because it compiles and then answers one work item's number.
        """
        if not names:
            return []
        magic_index:int = _magic_loop_index(body)
        if magic_index >= len(body) or not isinstance(body[magic_index], ast.For):
            # A gid-style body has no magic loop, so there is no per-item region to check.
            return []
        loop:ast.For = body[magic_index]
        offenders:List[ast.stmt] = []
        for stmt in loop.body:
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        offenders.append(node)
        return offenders
