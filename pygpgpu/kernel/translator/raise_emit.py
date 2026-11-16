from __future__ import annotations
import ast
import builtins
from typing import (List, Optional)


class _RaiseMixin:
    """`raise` and `assert`: the failure as data the host turns back into an exception.

    A device launch reports one status word, so a python exception cannot unwind out of a
    work item. The failure travels as *information* instead: the raising work item writes the
    id of its raise site into a launch-scoped slot and sets a stop flag, every other work
    item leaves at its next loop checkpoint, and the host rebuilds the class, the message and
    the line from the site table after the launch. See `pykernel_returns` for the host half.
    """

    # The two device-side locations a raise writes: the site id, and a flag every work item
    # checks so the rest of the launch stops doing work. Both are launch-scoped buffers the
    # render declares and the host reads back (see render_exception_params).
    EXCEPTION_SLOT:str = "_exception_site"
    EXCEPTION_STOP:str = "_exception_stop"
    # The text a raise leaves where its exit statement goes. A helper's return type is not
    # known when a raise is visited (it is settled from the body's `return`s, which may come
    # later), so the line is patched once the type is -- see `patch_raise_exits`.
    RAISE_EXIT_PLACEHOLDER:str = "@@pygpgpu_raise_exit@@"

    def _emit_exception_checkpoint(self)->None:
        """Stop this work item when another one has raised.

        The work items of a launch run independently, so a raise in one of them cannot end
        the others by itself. Each loop iteration therefore checks the launch-wide stop flag
        first: once any work item has raised, the rest leave at their next checkpoint instead
        of computing with the failed state. The check costs one load per iteration and is
        emitted only for a body that can raise.
        """
        if not self.has_raises:
            return
        self._emit(f"if ({self.EXCEPTION_STOP}[0])")
        self._emit("{")
        self._emit("    return;")
        self._emit("}")


    def _visit_raise(self, node:ast.Raise)->None:
        """`raise` as data the host turns back into python's exception.

        A work item cannot unwind into python: the launch reports one status word, and an
        abort threw away the exception's type, message and line. So the failure travels as
        *information*. The raising work item records the id of its raise site in the
        launch-scoped exception slot and sets the stop flag; every other work item leaves at
        the next checkpoint (the header of every loop), and the host rebuilds the exception
        from the site table after the launch.

        **The message must be a string literal**, and a computed one is refused here rather
        than carried. A raise identifies one place in the source, and that place's text is
        what says what went wrong; the device has no string type to carry a value-built
        message, so accepting one would silently lose it -- worse than refusing it with the
        reason. A bare `raise SomeError` (no message) is python-legal and supported.
        """
        class_name:str = "Exception"
        message:Optional[str] = None
        exc:Optional[ast.expr] = node.exc
        if exc is not None:
            if isinstance(exc, ast.Call):
                if exc.keywords:
                    self._error(
                        "raise with keyword arguments is not supported: give the exception "
                        "a single literal message, `raise ValueError(\"...\")`", node)
                callee:ast.expr = exc.func
                class_name = self._exception_class_name(callee, node)
                if exc.args:
                    if len(exc.args) > 1:
                        self._error(
                            "raise with several arguments is not supported: the device "
                            "carries one literal message, `raise ValueError(\"...\")`", node)
                    message = self._literal_exception_message(exc.args[0], node)
            else:
                class_name = self._exception_class_name(exc, node)
        site_id:int = len(self.raise_sites)
        self.raise_sites.append({"id": site_id, "class": class_name, "message": message,
                                 "line": getattr(node, "lineno", 0) or 0})
        self.has_raises = True
        self._emit(f"{self.EXCEPTION_SLOT}[0] = {self._raise_site_code(site_id)};")
        self._emit(f"{self.EXCEPTION_STOP}[0] = 1;")
        # Nothing after a raise in the same block can run: the work item ends here, and the
        # stop flag keeps the other work items from doing more work. A helper with a return
        # type must answer *something* on every path to compile, and the type may not be
        # known yet at the raise (a helper's return type is settled from its `return`s, which
        # can come later in the body) -- so the statement is a placeholder this translation
        # unit patches once the type is known (see `patch_raise_exits`).
        self._raise_exit_lines.append(len(self.lines))
        self._emit(self.RAISE_EXIT_PLACEHOLDER)
        self._emit_blank()


    def _raise_site_code(self, site_id:int)->str:
        """The value a raise writes into the slot: the site's id in the launch's table.

        A kernel numbers its sites from one, and the host's table is the kernel's own list.
        A helper's sites are adopted by the caller in call order (see `_adopt_raise_sites`),
        which is also the order the caller renders its callees, so the local id plus the
        caller's running total is exactly where the site lands. The caller therefore passes
        that total in as `EXCEPTION_BASE` and the helper adds its local id to it.
        """
        if getattr(self, "is_kernel", False):
            return str(site_id + 1)
        base:int = int(getattr(self, "raise_site_base", 0) or 0)
        return str(base + site_id + 1)


    def _raise_exit_statement(self)->str:
        """The statement that ends a work item at a raise, valid in a void or typed function."""
        return_type:Optional[object] = getattr(self, "return_type", None)
        if not return_type or return_type == "void":
            return "return;"
        if not isinstance(return_type, str):
            return "return;"
        return f"return ({self._device_type(return_type)})0;"


    def patch_raise_exits(self)->None:
        """Replace the raise-exit placeholders now that this unit's return type is known.

        Called once the body has been walked (a helper's return type is inferred from its
        `return`s, so it is not known while a raise is visited). Every placeholder becomes
        the `return` the enclosing function can legally contain: a bare `return;` for a void
        kernel or helper, and a zero of the declared type for one that answers a value --
        which the caller discards, because the host raises before reading any result.
        """
        if not self._raise_exit_lines:
            return
        statement:str = self._raise_exit_statement()
        self._replace_raise_exit_lines(self.lines, statement)


    def _replace_raise_exit_lines(self, lines:List[str], statement:str)->None:
        """Swap the placeholders in one line buffer for `statement`, keeping the indent.

        `_emit` prefixes the line with `self._indent` levels, so the comparison is on the
        stripped text and the indentation is carried over -- a placeholder inside a branch
        has to keep its own indentation in the replaced statement.
        """
        marker:str = self.RAISE_EXIT_PLACEHOLDER
        for index, line in enumerate(lines):
            if line.strip() != marker:
                continue
            indent:str = line[:len(line) - len(line.lstrip())]
            lines[index] = indent + statement
        self._raise_exit_lines = []
        """The statement that ends a work item at a raise, valid in a void or typed function."""
        return_type:Optional[str] = getattr(self, "return_type", None)
        if not return_type or return_type == "void":
            return "return;"
        return f"return ({self._device_type(return_type)})0;"


    def _visit_assert(self, node:ast.Assert)->None:
        """`assert` as a raise of `AssertionError`, which is what python makes of it.

        `assert cond, "text"` is `if not cond: raise AssertionError("text")`, so it takes
        exactly the same path: the failure is recorded in the exception slot and the host
        rebuilds python's `AssertionError` with that text.
        """
        condition:str = self._condition_code(node.test)
        self._emit(f"if (!({condition}))")
        self._emit("{")
        self._indent += 1
        try:
            message:Optional[str] = None
            if node.msg is not None:
                message = self._literal_exception_message(node.msg, node)
            site_id:int = len(self.raise_sites)
            self.raise_sites.append({"id": site_id, "class": "AssertionError",
                                     "message": message,
                                     "line": getattr(node, "lineno", 0) or 0})
            self.has_raises = True
            self._emit(f"{self.EXCEPTION_SLOT}[0] = {self._raise_site_code(site_id)};")
            self._emit(f"{self.EXCEPTION_STOP}[0] = 1;")
            self._emit("return;")
        finally:
            self._indent -= 1
        self._emit("}")
        self._emit_blank()


    def _exception_class_name(self, callee:ast.expr, node:ast.AST)->str:
        """The builtin exception class a `raise` names, checked against the real hierarchy.

        A raise of something that is not an exception is python's own `TypeError` at run
        time; refusing it at translation time says so where the user can see the line.
        """
        name:Optional[str] = None
        if isinstance(callee, ast.Name):
            name = callee.id
        elif isinstance(callee, ast.Attribute):
            name = callee.attr
        if name is None:
            self._error(
                "raise needs a builtin exception class: `raise ValueError(\"...\")`", node)
        candidate:object = getattr(builtins, name, None)
        if not (isinstance(candidate, type) and issubclass(candidate, BaseException)):
            self._error(
                f"raise {name} is not supported: it names no builtin exception class, so "
                f"the host has nothing to rebuild. Use one of the builtin exceptions "
                f"(ValueError, RuntimeError, AssertionError, ...)", node)
        return name


    def _literal_exception_message(self, expr:ast.expr, node:ast.AST)->Optional[str]:
        """A raise's message as fixed text, or a refusal saying why it cannot be carried.

        Only a plain string literal is accepted. A computed message (`"... " + str(v)`, an
        f-string with a field) needs a string in the device's type system, which the
        translator does not have; carrying the *site* is what lets the host rebuild the
        exception, and the site's text is the literal. Refusing here is the honest end: the
        alternative is a device launch whose message vanished.
        """
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if isinstance(expr, ast.Constant) and expr.value is None:
            return None
        if isinstance(expr, ast.JoinedStr) and all(
                isinstance(part, ast.Constant) and isinstance(part.value, str)
                for part in expr.values):
            # A field-free f-string is a literal in every way that matters.
            return "".join(part.value for part in expr.values)  # type: ignore[attr-defined]
        self._error(
            "a raise's message must be a literal string: the device carries the raise "
            "site, and the site's text is what the host puts in the exception. A message "
            "built from a value has no way to travel (the device has no string type), so "
            "write the text as a literal and put the value in a separate array", node)
        return None