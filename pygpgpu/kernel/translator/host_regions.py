from __future__ import annotations
import ast
from typing import (List, Optional, Set)
import numpy as np
from .notation import (
    DeviceType,
)
from .predicates import (
    infer_ctype,
    is_ptr,
)
from .walk import (
    _HOST_CREATION_FUNCTIONS,
    _NUMPY_DEVICE_TYPES,
    _UNRESOLVED,
    _assigned_name_of,
    _creation_target_of,
    _magic_loop_index,
)


class _HostRegionsMixin:

    def _bind_host_values(self, body:List[ast.stmt])->None:
        """Record every host-created object above the magic loop before the body is visited.

        The magic-loop scan needs to know which arrays exist -- `for i in
        np.ndindex(created.shape)` defines the iteration domain from a created array -- and
        every later use needs the object's type, so the declarations are collected once,
        up front, instead of being discovered while walking the body.
        """
        for stmt in body[:_magic_loop_index(body)]:
            value:Optional[ast.expr] = _creation_target_of(stmt)
            name:Optional[str] = _assigned_name_of(stmt)
            if value is None or name is None:
                continue
            if not self._is_creation_expression(value):
                continue
            produced:object = self.host_creation_values.get(name, _UNRESOLVED)
            if produced is _UNRESOLVED:
                self._error(
                    f"'{name}' cannot be created here: the expression only builds "
                    f"data, but it could not be evaluated on the host; build it from values "
                    f"that exist at call time (a shape, a constant, another array)",
                    value)
            device_type:DeviceType = infer_ctype(produced, self.type_map)
            self.host_values[name] = {"expr": value, "type": device_type}
            # The object travels as a parameter, appended after the outer-scope values
            # (see render_kernel and PyKernel._invoke_inner for the launch order).
            self.params.append((name, device_type))


    def _host_created_assignment(self, node:ast.stmt)->bool:
        """Skip one creation statement: it was already bound by _bind_host_values.

        Returns True when the statement was handled here (the device needs no code for it).
        Only a plain `name = <creation>` counts: a creation buried in an expression the
        device must evaluate is not a creation this can lift out.
        """
        name:Optional[str] = _assigned_name_of(node)
        if name is None:
            return False
        return name in self.host_values


    def _visit_host_statement(self, node:ast.stmt)->None:
        """Bind one host-region statement, or translate it when the host could not run it.

        A statement above the first magic loop (or between two of them) runs **once**, on
        the host, exactly as python reads it -- never once per work item. So when the
        interpreter already ran it (its line is in `host_statement_lines`, see
        `_host_region_values`), the device gets no code for it at all: only the values it
        bound travel to the kernel, as parameters.

        A statement the host could not run stays device code, and that is what keeps
        `local_array_per_group(...)` above a loop working: a shared buffer is declared by the
        kernel, so there is nothing for the host to execute.
        """
        if int(getattr(node, "lineno", 0) or 0) in self.host_statement_lines:
            self._bind_host_statement(node)
            return
        self._visit_statement(node)


    def _loop_written_names(self)->Set[str]:
        """Names the magic loops' bodies assign as a plain python local.

        Such a name is a plain local the work item updates; only one that is *also*
        initialized above the loop and read after it is carried through a device slot (see
        `_carried_scalars`). The distinction matters: the names here are what keeps a host
        scalar from being bound twice, so this is deliberately narrower than "any name the
        loop assigns" -- a `self.<attr> += ...` or an element store is not a local at all.
        """
        body:List[ast.stmt] = list(self.tree.body)
        start:int = _magic_loop_index(body)
        names:Set[str] = set()
        for statement in body[start:]:
            if not isinstance(statement, ast.For):
                continue
            for child in ast.walk(statement):
                if isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
                    names.add(child.target.id)
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
        return names


    def _bind_host_statement(self, node:ast.stmt)->None:
        """Give the device the values a host statement bound.

        The statement ran on the host (see `_host_region_values`), so its value exists at
        launch time; what is recorded here is which of its names the device has to receive.
        A name a device argument can carry becomes a scalar or array parameter. A name it
        cannot carry -- a string, a dict, an arbitrary-precision number -- stays on the
        host: nothing is bound for the device, and no statement that reads it can be device
        code (that was decided when the host ran it). A statement that binds no value
        (`import json`) contributes neither.
        """
        # `lo, hi = bounds(1)` above the loop: the host ran the helper, so both values are
        # known, and each name becomes its own scalar parameter. The device never calls the
        # helper -- the pair is data by the time the launch happens.
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Tuple):
            for element in node.targets[0].elts:
                if isinstance(element, ast.Name):
                    self._bind_host_value(element.id, element)
            return
        name:Optional[str] = _assigned_name_of(node)
        if name is None or name in self.host_values:
            return
        # An outer-scope name (`nonlocal held` in an enclosing `def`, or `global`) is *not*
        # a host value: it travels through its own slot (see _bind_outer_variables), and
        # every use of it already compiles against that. Treating it as a host scalar
        # declared it twice and gave the device a pointer where the outer slot is a value.
        if name in self.outer_vars:
            return
        # A name the *loop* updates (`total = 0.0` above it, `total = total + x[i]` inside)
        # is one location for the whole launch, carried through a device slot (see
        # _bind_carried_scalars). Its initializer is host work in the sense that python ran
        # it, but the slot is what carries the value, and the kernel is what finalizes it.
        # A `return total` *inside* the loop does not qualify as a carried scalar (the
        # helper requires the name to be read after the loop), so the loop body is
        # consulted directly.
        if name in self.carried_scalars or name in self._loop_written_names():
            # Not a host value: the loop updates it, so the ordinary walk owns the
            # initializer (a device local, or a carried slot). Translating it here is what
            # keeps the work item's own copy declared -- and it is what the body meant.
            self._visit_statement(node)
            return
        # `part = a[0:4]` is a pointer into the array, not a host value (see
        # _emit_array_slice): the device binds the offset itself.
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript) \
                and isinstance(node.value.slice, ast.Slice):
            return
        # `weights = [1.0, 2.0, 3.0]` feeding `enumerate(weights)` is a compile-time table:
        # the loop reads the elements from the list's own literal (see
        # _emit_constant_enumerate_head), so the binding emits no device code and no
        # parameter carries it.
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            assigned:Optional[str] = _assigned_name_of(node)
            if assigned is not None and assigned in self._constant_list_names:
                return
        # `src = a` is a pointer assignment, not a host value: no parameter carries the
        # local, and letting it become a host scalar made the launch look for an extent the
        # argument list does not have (an `IndexError` from the shape check).
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            source_name:Optional[str] = self._pointer_source_name(node.value)
            if source_name is not None and is_ptr(self._lookup(source_name)):
                self._visit_statement(node)
                return
        produced:object = self.host_region_values.get(name, _UNRESOLVED)
        if produced is _UNRESOLVED:
            # The host ran the statement but bound no value under this name, so there is
            # nothing for the device to receive; the name simply stays host-side.
            return
        self._bind_host_value(name, node)


    def _bind_host_value(self, name:str, node:Optional[ast.AST])->None:
        """Give the device one host-bound name, as a scalar (or array) parameter.

        The value came from a statement the interpreter ran (see `_host_region_values`), so
        it exists at launch time. A type a device argument can carry becomes a parameter; a
        string, a dict or an arbitrary-precision number cannot, and stays on the host.
        """
        if name in self.host_values or name in self.host_scalars:
            return
        produced:object = self.host_region_values.get(name, _UNRESOLVED)
        if produced is _UNRESOLVED:
            return
        try:
            device_type:DeviceType = infer_ctype(produced, self.type_map)
        except SyntaxError:
            device_type = None
        if device_type is None:
            # Not something a device argument can carry: the value stays on the host.
            return
        # A *host value* is its own entry list, not an entry of host_values: host_values is
        # what the body *created* (see _bind_host_values), and the launch treats those
        # differently -- a creation is passed as an object, this one as a scalar.
        self.host_scalars[name] = device_type
        self.params.append((name, device_type))



    def _is_creation_expression(self, node:ast.expr)->bool:
        """True when this expression *builds data* the device cannot build itself.

        A plain constant is ordinary python that the device handles directly, so it is not
        a creation: `total = 0.0` stays a local variable.
        """
        if isinstance(node, ast.Constant):
            return False
        if isinstance(node, ast.Name):
            return False
        if isinstance(node, ast.Attribute):
            # np.float32 and friends: a dtype/type object, not a value.
            if isinstance(node.value, ast.Name) and self.resolve_global(node.value.id) is np:
                return True
            return False
        if isinstance(node, ast.Call):
            func:ast.expr = node.func
            if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and self.resolve_global(func.value.id) is np):
                return False
            if self._is_numpy_dtype_name(func.attr):
                # `np.float32(2.5)` builds a constant the host can hold, while
                # `np.float32(x[i] * 2.0)` converts a device value in place (see
                # _emit_numpy_cast). Only the constant form is a creation, so a cast inside
                # the loop keeps working instead of being refused as a per-work-item one.
                return all(isinstance(arg, ast.Constant) for arg in node.args)
            return self._is_creation_function(func.attr)
        return False

    def _is_numpy_dtype_name(self, name:str)->bool:
        """True for a numpy scalar type (`np.float32`, `np.int32`, ...)."""
        return name in _NUMPY_DEVICE_TYPES or name in self.RETURN_SLOT_TYPES


    def _is_numpy_constant_cast(self, node:ast.expr)->bool:
        """True for `np.<dtype>(<constants>)`: one number, known at translation time.

        The creation guard treats a constant dtype call as a creation (see
        `_is_creation_expression`), and above the magic loop that is right -- the host
        builds the value and it travels as a parameter. Below the loop the value is one
        per work item, and a scalar is exactly that: the device cast computes the same
        number python would, so folding it to a constant keeps the statement a plain
        assignment instead of refusing a cast as if it allocated.
        """
        if not isinstance(node, ast.Call):
            return False
        func:ast.expr = node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and self.resolve_global(func.value.id) is np):
            return False
        if not self._is_numpy_dtype_name(func.attr):
            return False
        return all(isinstance(arg, ast.Constant) for arg in node.args)


    def _is_creation_function(self, name:str)->bool:
        """A numpy function that only allocates or reshapes host data."""
        if name in _HOST_CREATION_FUNCTIONS:
            return True
        return name.endswith("_like")


    def _visit_work_item_body(self, statements:List[ast.stmt])->None:
        """Visit the magic loop's body: these statements run once per work item.

        A `return` here is exactly what a kernel return is (one work item leaving early),
        so this is *not* the "after the loop" region -- but a creation here would build one
        object per work item, which is why creations are refused with `after_loop_only`.

        `in_work_item_body` is set for the duration: a store to a launch-wide location here
        is one location shared by every work item, which is a race unless it is atomic or
        guarded (see `_named_outer_target`). It is deliberately **not** `after_magic_loop`:
        that flag means "runs after the dispatch", which a statement inside the loop does
        not, and setting it there refused a perfectly good `return` from inside the loop.
        """
        previous:bool = self.after_magic_loop
        previous_creation:bool = self.creation_below_loop
        previous_body:bool = self.in_work_item_body
        self.after_magic_loop = previous
        self.creation_below_loop = True
        self.in_work_item_body = True
        try:
            for stmt in statements:
                self._visit_statement(stmt)
        finally:
            self.after_magic_loop = previous
            self.creation_below_loop = previous_creation
            self.in_work_item_body = previous_body


    def _store_is_guarded(self, node:ast.AST)->bool:
        """Whether a store is inside a branch every work item takes the same way.

        A template for one work item's slice is a real pattern (`if i == 0: total = x[0]`),
        and its store is not racy because exactly one work item enters the branch. The
        condition has to be a test on a *launch-wide* value -- the work-item index, a scalar
        argument, an outer scalar -- so a data-driven condition (`if x[i] > 0`) stays refused:
        there every work item decides for itself, which is the race.
        """
        for guard in self._guard_conditions:
            names:List[str] = [n.id for n in ast.walk(guard)
                               if isinstance(n, ast.Name)]
            if not names:
                continue
            if all(self._is_launch_wide_name(name) for name in names):
                return True
        return False


    def _visit_loop_exit(self, node:ast.stmt, spelling:str)->None:
        """Translate `break`/`continue` for the loop they actually sit in.

        Three cases, and the difference between them is not cosmetic:

        * **Inside a real loop** (a nested serial `for`/`while`, or the body of an element
          loop): the generated source has a C loop there, so the statement is emitted
          verbatim and keeps python's meaning.
        * **Directly in a work item** (the magic loop's body, which becomes the dispatch
          rather than a loop): `continue` means "this work item stops here and the next one
          -- which is another thread -- does its own work", so it becomes a plain `return;`.
          That is the *same* control flow, because there is no loop left to continue. It must
          NOT go through _visit_kernel_return: that one claims the kernel result, so a work
          item merely skipping its turn would race the value-returning sites for the answer
          and could hand back None.
        * **`break` in a work item**: there is no meaning to give it. Stopping the magic
          loop means stopping every other work item too, and they are already running in
          parallel, so the only honest translation is a refusal.
        """
        if self.loop_depth:
            self._emit(f"{spelling};")
            return
        if not self.is_kernel or not self.creation_below_loop:
            # Outside a loop and outside a work item: nothing to exit and nothing to
            # enforce. `break`/`continue` outside a loop is a SyntaxError in python too,
            # and the body's own statements above the loop never reach here as device code.
            self._emit(f"{spelling};")
            return
        if spelling == "continue":
            self._emit("return;")
            return
        self._error(
            "`break` cannot stop a magic loop: its iterations are work items running in "
            "parallel, so stopping the loop means stopping every other work item too, which "
            "the device cannot express. Guard the rest of this work item's body with `if`, "
            "or put the sequential loop you want to leave inside the body with "
            "`for j in range(...)` and `break` that one instead", node)
