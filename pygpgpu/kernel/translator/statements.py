from __future__ import annotations
import ast
import re
from typing import (List, Optional)
import numpy as np
from ..structs import dtype_for as structs_dtype_for
from .notation import (
    DeviceType,
    Translated,
)
from .builtins import (
    _translation_branch,
)
from .predicates import (
    is_int,
    is_ptr,
)


class _StatementsMixin:

    def _visit_statement(self, node:ast.stmt)->None:
        # A creation statement was executed by the interpreter and travels as a parameter,
        # so the device emits nothing for it. Checked here as well as in the magic-loop
        # header: a gid-style body has no loop above which to lift them, yet its top-level
        # statements are just as much "once per launch".
        if self.is_kernel and self._host_created_assignment(node):
            return
        if isinstance(node, ast.Assign):
            # An initializer for a carried scalar already ran on the host, and the slot that
            # travels is the buffer it filled: re-emitting 	otal = 0 would overwrite the
            # parameter's type and reset the shared location for every work item.
            target:object = node.targets[0] if len(node.targets) == 1 else None
            if (isinstance(target, ast.Name) and target.id in self.carried_scalars
                    and not self.creation_below_loop):
                return
            self._visit_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._visit_augassign(node)
        elif isinstance(node, ast.If):
            # `if KERNEL_TRANSLATING:` / `if not KERNEL_TRANSLATING:` is a *compile-time*
            # branch, the translator's counterpart of `typing.TYPE_CHECKING`: the constant is
            # False in python (so the interpreter backend runs the python branch, which it
            # does by executing the body as written) and True here, so exactly one branch is
            # translated and the other is dropped -- a macro, not a runtime test. Any other
            # condition is an ordinary device `if`.
            taken:Optional[List[ast.stmt]] = _translation_branch(node.test, node.body,
                                                                 node.orelse)
            if taken is not None:
                self._emit_block(taken)
                return
            cond:str = self._condition_code(node.test)
            # `_condition_code` already wraps the condition in the one pair C's syntax needs
            # -- it has to, because the boolean join and the ternary below splice it into a
            # larger expression. Emitting `if ({cond})` therefore produced `if ((n == k))`,
            # and the OpenCL compiler reported the inner pair as an extraneous one around a
            # comparison (with a `use '=' ...` hint that would change the meaning).
            self._emit(f"if {cond}")
            self._emit("{")
            self._guard_conditions.append(node.test)
            try:
                self._push_scope_block(node.body)
            finally:
                self._guard_conditions.pop()
            if node.orelse:
                self._emit("}")
                self._emit("else")
                self._emit("{")
                self._push_scope_block(node.orelse)
            self._emit("}")
            self._emit_blank()
        elif isinstance(node, ast.While):
            cond:str = self._condition_code(node.test)
            # Same as the `if` above: the condition arrives with its own pair.
            self._emit(f"while {cond}")
            self._emit("{")
            self._push_scope_block(node.body, is_loop=True)
            self._emit("}")
            self._emit_blank()
        elif isinstance(node, ast.For):
            # A second *top-level* magic loop would silently become an ordinary serial loop
            # inside every work item -- each item running the whole phase, all of them racing
            # on the same array. That is a different computation, so it is refused here
            # rather than emitted; splitting the body across several @kernel calls is what
            # this means. A `for` inside the work-item body (`creation_below_loop`) is an
            # ordinary loop and stays allowed.
            if (self.is_kernel and not self.creation_below_loop
                    and self._is_secondary_magic_loop(node)):
                self._error(
                    "this kernel body has more than one magic loop, and only the first "
                    "becomes the work-item dispatch: a second one would run inside every "
                    "work item instead of once, so several work items would repeat the same "
                    "phase and race on the same array. Give each phase its own @kernel and "
                    "call them in order, which keeps the data on the device", node)
            self._visit_for(node)
        elif isinstance(node, ast.Return):
            self._visit_return(node)
        elif isinstance(node, ast.Expr):
            # Bare string literals (e.g. the function docstring) are ignored, not emitted into the kernel
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                pass
            elif self._is_print_call(node.value):
                # print(...) maps to device printf calls that print the same text
                self._emit_print(node.value)
            else:
                # Other expression statements (e.g. gpu.atomic_add(...)/gpu.barrier()) must be emitted, otherwise side effects are lost
                code, _ = self._visit_expression(node.value)
                if code:
                    self._emit(f"{code};")
        elif isinstance(node, ast.AnnAssign):
            # `x: float = 0.0` is ordinary Python. The annotation has no device meaning --
            # the type comes from the assigned value, as it does for every other name -- so
            # it is dropped and the statement becomes a plain assignment. A bare `x: float`
            # declares nothing, which a kernel has no use for, so it is ignored the way a
            # docstring is.
            if node.value is not None:
                self._visit_assign(ast.Assign(targets=[node.target], value=node.value,
                                              lineno=node.lineno,
                                              col_offset=node.col_offset))
        elif isinstance(node, ast.Break):
            self._visit_loop_exit(node, "break")
        elif isinstance(node, ast.Continue):
            self._visit_loop_exit(node, "continue")
        elif isinstance(node, (ast.Nonlocal, ast.Global)):
            # The scope declarations themselves emit nothing: they were already read while
            # binding the outer variables (see _bind_outer_variables), so the name travels
            # as a device parameter and every use goes through it.
            pass
        elif isinstance(node, ast.Pass):
            pass
        else:
            self._error(f"unsupported statement: {node.__class__.__name__}", node)


    def _emit_block(self, statements:List[ast.stmt])->None:
        """Translate statements as a plain block, with a scope of their own.

        Used by the compile-time branch (`if KERNEL_TRANSLATING`), which contributes its
        statements *inline*: it is a macro, so no `if` and no braces survive into the
        generated source.
        """
        self.scopes.append({})
        for statement in statements:
            self._visit_statement(statement)
        self.scopes.pop()


    def _push_scope_block(self, statements:List[ast.stmt], is_loop:bool=False)->None:
        """Translate a braced block, optionally marking it as a real C loop body.

        `is_loop` is what makes `break`/`continue` mean their C meaning inside this block:
        the generated source emits them verbatim, and only a real loop can contain them
        (see _visit_loop_exit). A plain block -- the `if` half of an `if`/`else` -- leaves
        the depth alone, so an exit statement inside it belongs to whichever enclosing
        loop there is.
        """
        self.scopes.append({})
        self._indent += 1
        if is_loop:
            self.loop_depth += 1
        try:
            for stmt in statements:
                self._visit_statement(stmt)
        finally:
            if is_loop:
                self.loop_depth -= 1
            self._indent -= 1
            self.scopes.pop()


    def _visit_for(self, node:ast.For)->None:
        if isinstance(node.iter, (ast.Name, ast.Attribute)):
            # `for x in arr`: an element-wise walk becomes an indexed one, with `x` bound to
            # the current element by value -- the same read-only snapshot `enumerate` gives.
            self._emit_element_loop(node)
            return
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "enumerate"):
            self._error(
                "enumerate() is only supported as the outermost magic loop "
                "(`for i, x in enumerate(array)`), not as a real inner loop", node.iter)
        if (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr in ("ndindex", "ndenumerate")
                and isinstance(node.iter.func.value, ast.Name)
                and self.resolve_global(node.iter.func.value.id) is np):
            self._error(
                f"np.{node.iter.func.attr} is only supported as the outermost magic loop, "
                f"not as a real inner loop", node.iter)
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"):
            self._error("only `for ... in range(...)` loops are supported", node.iter)
        args:List[ast.expr] = node.iter.args
        if len(args) not in (1, 2, 3) or node.iter.keywords:
            self._error("range() takes 1-3 positional arguments", node.iter)
        if not isinstance(node.target, ast.Name):
            self._error("loop target must be a plain variable", node.target)

        values:List[Translated] = [self._visit_expression(a) for a in args]
        for a, (_, t) in zip(args, values):
            if not is_int(t):
                self._error("range() arguments must be integers", a)
        target:str = node.target.id
        if self._lookup(target) is None:
            # Python scopes the loop target to the whole function, and a C `for (int k ...)`
            # scopes it to the loop: a read after the loop named an undefined identifier. The
            # declaration is hoisted, and the loop drives a *hidden* counter that assigns the
            # target on each pass. The counter is what the exit increment belongs to, so the
            # target is left holding the last value the iterator yielded -- `range(3)` leaves
            # `k == 2` in python, and `for (k = 0; k < 3; k++)` would have left 3. It is also
            # what lets a body write the target without steering the loop: python rebinds it
            # from the iterator each pass, and a shared counter cannot tell the two apart.
            self._hoist_declaration(target, "int")
        self._define(target, "int", node.target)
        counter:str = f"_{target}_i"

        if len(values) == 1:
            start, stop, step = "0", values[0][0], None
        elif len(values) == 2:
            start, stop, step = values[0][0], values[1][0], None
        else:
            start, stop, step = values[0][0], values[1][0], values[2][0]

        if step is None:
            self._emit(f"for (int {counter} = {start}; {counter} < {stop}; {counter}++)")
        elif re.fullmatch(r"-?\d+", step):
            cond_op:str = (">" if step.startswith("-") else "<")
            self._emit(f"for (int {counter} = {start}; {counter} {cond_op} {stop}; {counter} += {step})")
        else:
            self._error("range() step must be a constant integer", node.iter.args[-1])
        self._emit("{")
        self._indent += 1
        try:
            self._emit(f"{target} = {counter};")
        finally:
            self._indent -= 1
        self._push_scope_block(node.body, is_loop=True)
        self._emit("}")
        self._emit_blank()


    def _emit_element_loop(self, node:ast.For)->None:
        """`for x in arr:` as an indexed loop with `x` a copy of the current element.

        A work item cannot write through an iterator, so the element is read-only -- which is
        what python's iteration gives for a record too. The index variable is hidden, so a
        body that also declares `i` does not collide with it.
        """
        if not isinstance(node.target, ast.Name):
            self._error("iterating an array needs a plain variable: `for x in arr`", node.target)
        code, t = self._visit_expression(node.iter)
        if not is_ptr(t):
            self._error(
                "iterating a value directly needs an array; this is a scalar. Use an array "
                "argument, or a range() loop", node.iter)
        element_t:object = t[1]
        holder:ast.expr = node.iter
        # The extent: a parameter's own shape variable, or a structure member's fixed length.
        if isinstance(holder, ast.Name):
            stop:str = self._shape_var(holder.id, 0)
        elif isinstance(holder, ast.Attribute) and isinstance(holder.value, ast.Name):
            base_t:DeviceType = self._lookup(holder.value.id)
            base_element:object = base_t[1] if is_ptr(base_t) else base_t
            dtype:Optional[np.dtype] = (structs_dtype_for(base_element)
                                        if isinstance(base_element, str) else None)
            if dtype is None or holder.attr not in (dtype.names or ()):
                self._error(
                    f"cannot iterate '{ast.unparse(holder)}': it is not an array member",
                    holder)
            field_dtype:np.dtype = np.dtype(dtype.fields[holder.attr][0])
            if field_dtype.subdtype is None:
                self._error(
                    f"cannot iterate '{ast.unparse(holder)}': a scalar member is not an array",
                    holder)
            stop = str(field_dtype.subdtype[1][0])
        else:
            self._error(
                f"cannot iterate '{ast.unparse(holder)}': use a named array parameter or an "
                f"array member of one", node.iter)

        name:str = node.target.id
        index:str = f"_{name}_idx"
        fresh:bool = self._lookup(name) is None
        if fresh:
            # The element copy is python's loop target: it survives the loop, so its
            # declaration is hoisted like any other first assignment (the type decides how the
            # zero is spelled, which is what a record element needs).
            self._hoist_declaration(name, self._device_type(element_t))
        self._emit(f"for (int {index} = 0; {index} < {stop}; {index}++)")
        self._emit("{")
        self._define(name, element_t, node.target)
        self._indent += 1
        self.loop_depth += 1
        try:
            # A first binding reads the hoisted declaration; a rebinding keeps the loop
            # body's own declaration, which shadows the outer name the way a fresh C
            # variable does and terminates the way the nested loop needs.
            self._emit(f"{name} = {code}[{index}];" if fresh
                       else f"{self._device_type(element_t)} {name} = {code}[{index}];")
            for statement in node.body:
                self._visit_statement(statement)
        finally:
            self.loop_depth -= 1
            self._indent -= 1
        self._emit("}")
        self._emit_blank()
