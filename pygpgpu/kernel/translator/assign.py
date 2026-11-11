from __future__ import annotations
import ast
from typing import (Dict, List, Optional, cast)
import numpy as np
from .private_array import _PrivateArrayMixin
from .type_map import (
    DTYPE_MAP,
)
from .notation import (
    BLOCK_LOCAL_BUILTINS,
    DeviceType,
    PER_GROUP_BUILTINS,
    WORK_GROUP_VALUES,
    Translated,
    device_name,
)
from .predicates import (
    is_image,
    is_ptr,
)
from .work_group import _WorkGroupMixin


class _AssignMixin(_PrivateArrayMixin, _WorkGroupMixin):

    def _visit_assign(self, node:ast.Assign)->None:
        if len(node.targets) != 1:
            self._error("chained assignment is not supported", node)
        target:ast.expr = node.targets[0]

        # `weights = [1.0, 2.0, 3.0]` feeding `enumerate(weights)` is a compile-time table:
        # the loop reads the elements from the literal itself (see
        # _emit_constant_enumerate_head), so the binding emits nothing and no parameter
        # carries it. Handled here rather than only where a host region binds it, because
        # both paths reach this method.
        if isinstance(target, ast.Name) and target.id in self._constant_list_names:
            return

        # `src = a` is a plain pointer assignment: an array parameter *is* a `T*` on the
        # device, so the work item binds a second name to the same buffer and reads or
        # writes through either. The statement itself is emitted by the ordinary assignment
        # path below; what is recorded here is which array it was assigned from, because the
        # *extents* are launch parameters named after the array -- a shape read through the
        # local (`src.shape[0]`) has to resolve to them, or the launch looks for an argument
        # that does not exist (an `IndexError` from the shape check). Only a bare name or a
        # receiver member is a candidate: `threadIdx.x` is an attribute too, and it is a
        # value, not an array.
        if isinstance(target, ast.Name) and self._lookup(target.id) is None:
            source_name:Optional[str] = self._pointer_source_name(node.value)
            if source_name is not None and is_ptr(self._lookup(source_name)):
                self._array_pointer_sources[target.id] = source_name

        # A *slice* of an array (`part = a[lo:hi]`) is a pointer into the same buffer: the
        # device has no view type, so the slice becomes `T* part = a + lo;` and its length
        # is the one the slice wrote down. A constant slice is a fixed extent, which is what
        # makes `part.shape[0]` a compile-time number (see _shape_var).
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Subscript) \
                and isinstance(node.value.slice, ast.Slice) \
                and self._lookup(target.id) is None:
            if self._emit_array_slice(target.id, node):
                return

        # `box.total = v` is a call to the property's setter. It is handled here rather
        # than in _visit_target because a setter takes the value, which only the
        # assignment knows -- and because the result is a statement, not an expression.
        if isinstance(target, ast.Attribute):
            base_code, base_t = self._visit_expression(target.value)
            if isinstance(base_t, tuple) and is_ptr(base_t):
                from .. import structs as _structs_module  # noqa: PLC0415

                dtype:Optional[np.dtype] = _structs_module.dtype_for(base_t[1])
                if dtype is not None and self._emit_property_write(
                        _structs_module.declared_classes(dtype), target.attr,
                        (base_code, base_t), node.value, node):
                    return

        # A creation below the magic loop runs once per work item, so its object cannot be a
        # host parameter. When the shape is a compile-time constant it is exactly a C local
        # array -- private storage per work item, which is what python's fresh array means --
        # and it is emitted as one. A runtime extent has no fixed-size declaration, so it is
        # refused instead of being silently shared between work items.
        if (self.creation_below_loop and isinstance(target, ast.Name)
                and self._is_creation_expression(node.value)):
            # A *scalar* dtype cast with constant arguments (`two = np.float64(2.0)`) is
            # not an allocation: python answers one number here, exactly as it does above
            # the loop, where the same statement is lifted to the host and travels as a
            # parameter. Emitting it as a device cast instead answers the same value with
            # no launch machinery -- the cast spelling is a cast wherever it appears. The
            # refusal used to trip on the dtype names sitting on the creation list, which
            # made one spelling answer three ways depending on where it appeared.
            if self._is_numpy_constant_cast(node.value):
                # The device spelling is the cast itself (`(double)(2.0)`), which answers
                # the same number python computes. Rebuilding the assignment would meet
                # the same creation guard again, so the store is emitted directly.
                func_attr:str = cast("str", node.value.func.attr)  # type: ignore[union-attr]
                element_code:str = self._emit_numpy_cast(
                    func_attr,
                    [(self._visit_expression(a)[0], "int") for a in node.value.args],
                    node.value)[0]
                element_t:object = DTYPE_MAP[func_attr]
                self._define(target.id, element_t, node)
                self._emit(f"{self._device_type(element_t)} {device_name(target.id)}"
                           f" = {element_code};")
                return
            if self._emit_private_array(target.id, node):
                return
            self._error(
                "a value cannot be created below the magic loop unless its shape is a "
                "compile-time constant: the statements there run once per work item, so a "
                "constant shape becomes a private array and anything else would create one "
                "object per work item. Create it above the loop, or give it a constant "
                "shape (e.g. np.zeros(4, dtype=np.float32))", node)

        # self.xxx = ... : a scalar member of the receiver structure is a field of the
        # pointer, so the store lands in the caller's structure -- the shape
        # `HitRecord.set_face_normal` needs. A host attribute (an ndarray, or the receiver's
        # own state) is still refused: device code has nowhere to write it back from.
        if (self.self_param and isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == self.self_param):
            from .. import structs as _structs_module  # noqa: PLC0415

            member:str = self.self_struct_members.get(target.attr, target.attr)
            receiver_t:DeviceType = self._lookup(self.self_param)
            receiver_dtype:Optional[np.dtype] = (
                _structs_module.dtype_for(receiver_t[1])
                if isinstance(receiver_t, tuple) and is_ptr(receiver_t) else None)
            is_member:bool = (receiver_dtype is not None
                              and member in _structs_module.member_types(
                                  receiver_dtype, self.backend))
            if is_member:
                value_code, value_t = self._visit_expression(node.value)
                if np.dtype(receiver_dtype.fields[member][0]).name == "bool":
                    # A bool member is a byte on the device: a float assigned straight into
                    # it truncates (`0.99` became 0), and the branch reading it went the
                    # wrong way -- the sphere lost its blue channel and came out grey.
                    value_code = f"(({value_code}) != 0)"
                self.member_writes.add(self.self_param)
                self._emit(f"({self.self_param})->{member} = {value_code};")
                return
            self._error(
                f"cannot assign to self.{target.attr}: host attributes cannot be written by device code; "
                f"write into an array element (self.{target.attr}[i] = ...) or use a local variable", node)

        # shared memory declaration: buf = local_array_per_group(64, np.float32)
        if (isinstance(node.value, ast.Call)
                and self._builtin_call_name(node.value) in (
                    PER_GROUP_BUILTINS + BLOCK_LOCAL_BUILTINS)):
            self._handle_local_declaration(node)
            return

        # Tuple unpacking/swap: a, b = b, a -- also when an item is an array element,
        # as in the idiomatic swap a[i], b[i] = b[i], a[i].
        if isinstance(target, ast.Tuple):
            # `x, y = pair(...)`, where the helper answers a tuple: python unpacks the
            # pair, and the device does the same by passing the destinations as the
            # helper's output parameters (see _visit_return's tuple branch, which
            # stores through them). Any other non-tuple right-hand side stays refused.
            if not isinstance(node.value, ast.Tuple):
                if isinstance(node.value, ast.Call) \
                        and self._tuple_call_element_count(node.value) is not None:
                    count:int = cast(int, self._tuple_call_element_count(node.value))
                    if len(target.elts) != count:
                        self._error(
                            f"the call answers {count} values, and the tuple assignment "
                            f"unpacks {len(target.elts)} names -- the counts have to "
                            f"match, as they do in python", node)
                    self._emit_tuple_call_unpack(target.elts, node.value)
                    return
                self._error("tuple assignment requires matching tuples on both sides", node)
            if not isinstance(node.value, ast.Tuple) or len(target.elts) != len(node.value.elts):
                self._error("tuple assignment requires matching tuples on both sides", node)
            if not all(isinstance(t, (ast.Name, ast.Subscript)) for t in target.elts):
                self._error(
                    "tuple assignment targets must be plain variables or array elements",
                    node)

            values:List[Translated] = [self._visit_expression(v) for v in node.value.elts]
            temps:List[Translated] = []
            for code, t in values:
                if is_ptr(t):
                    self._error("cannot swap array parameters", node)
                self._tmp_counter += 1
                tmp:str = f"_swap{self._tmp_counter}"
                self._define(tmp, t, node)
                self._emit(f"{self._device_type(t)} {tmp} = {code};")
                temps.append((tmp, t))

            for (tmp, t), target_item in zip(temps, target.elts):
                if isinstance(target_item, ast.Subscript):
                    # Evaluate the destination after every right-hand side has been
                    # captured, so a swap really reads both values first.
                    code, _ = self._visit_array_target(target_item.value)
                    index_code, _ = self._visit_subscript_index(target_item)
                    self._emit(f"{code}[{index_code}] = {tmp};")
                    continue
                name:str = target_item.id
                existing:DeviceType = self._lookup(name)
                if existing is not None and is_ptr(existing):
                    self._error(f"cannot assign to array parameter '{name}'", node)
                is_new:bool = (existing is None)
                self._define(name, t, node)
                decl:str = (f"{self._device_type(t)} " if is_new else "")
                self._emit(f"{decl}{name} = {tmp};")
            return
        code, t = self._visit_expression(node.value)
        from .. import structs as _structs_module

        if is_ptr(t) and not (isinstance(t, tuple) and _structs_module.dtype_for(t[1]) is not None):
            # A pointer to a *structure* is an aliasable handle (`q = item; q.a` counts as a
            # read of `item.a`), so it copies like a scalar. An array pointer is the same
            # thing on the device -- an array parameter *is* a `T*` -- so `src = a` binds a
            # second name to the same buffer and the work item reads/writes it through
            # either. Rebinding an *outer* array stays refused (the object is shared with the
            # caller, so only its elements may be written), and that is the one case left.
            if isinstance(target, ast.Name) and target.id in self.outer_vars:
                self._error(
                    f"cannot rebind the outer array '{target.id}': the array object itself "
                    f"is shared with the caller, so a device kernel can only write its elements", node)
            if isinstance(target, ast.Name) and self._lookup(target.id) is None:
                # One pointer assigned to a fresh local: a real C declaration, so every later
                # use of the name is an ordinary variable read.
                self._emit(f"{self._device_type(t[1])}* {device_name(target.id)} = {code};")
                self._define(target.id, t, node)
                return
            self._error("cannot assign an array to a variable", node)

        if isinstance(target, ast.Name):
            outer:Optional[Dict[str, object]] = self._named_outer_target(target, node)
            if outer is not None:
                self._emit(f"{device_name(target.id)}[0] = {code};")
                return
            # `rows = out.shape[0]`: the name stands for the array's extent, which is
            # what an `np.ndindex(rows, ...)` bound resolves through (see
            # _extent_source). Recorded before the visit of the loop, whose head reads
            # the table.
            if isinstance(node.value, ast.Subscript) \
                    and isinstance(node.value.slice, ast.Constant) \
                    and isinstance(node.value.slice.value, int):
                holder:Optional[str] = self._shape_holder_param(node.value.value)
                if holder is not None:
                    holder_t:DeviceType = self._lookup(holder)
                    if is_ptr(holder_t) or is_image(holder_t):
                        self._shape_aliases[target.id] = (holder,
                                                          int(node.value.slice.value))
            existing:DeviceType = self._lookup(target.id)
            if isinstance(existing, tuple) and len(existing) == 2 and existing[0] == "carried":
                # One location for the whole launch: store through the pointer, exactly as
                # the outer-scalar path does.
                self._emit(f"{device_name(target.id)}[0] = {code};")
                return
            if existing is not None and is_ptr(existing):
                self._error(f"cannot assign to array parameter '{target.id}'", node)
            is_new:bool = (existing is None)
            # A local holding the block id stays marked as one, so `buf[blk]` later still
            # reads as a group selection rather than an ordinary index (see
            # _is_block_id_expression).
            stored:object = t
            if self._is_block_id_expression(node.value):
                stored = "blockid"
            self._define(target.id, stored, node)
            decl:str = ""
            if is_new:
                if isinstance(t, tuple) and is_ptr(t):
                    # A structure pointer alias: declared with the *same* const qualification
                    # the source expression has, because a `const Struct*` parameter cannot
                    # initialise a mutable `Struct*`.
                    const:str = ("const " if isinstance(node.value, ast.Name)
                                 and self._is_const_struct(node.value.id) else "")
                    decl = f"{const}{self._kernel_scalar_type(t[1])}* "
                else:
                    decl = self._device_type(t) + " "
                if len(self.scopes) > 1:
                    # The assignment is inside a branch or loop, so this block is where the
                    # declaration would land -- and a read after the block would then have no
                    # declaration in scope. Python scopes the name to the whole function, so
                    # the declaration moves to the top of it and this line keeps only the
                    # store. The value is zero: a read between the block and a later
                    # assignment sees 0 rather than an uninitialised register, which is the
                    # closest a device gets to python's UnboundLocalError.
                    self._hoist_declaration(target.id, decl.strip())
                    self._emit(f"{device_name(target.id)} = {code};")
                    return
                # A later widening (`scale = 1.0` here, a double inside the loop) has to
                # rewrite this declaration's type -- see _render_body's fixup pass.
                self._declaration_sites.append((len(self.lines), target.id,
                                                self._device_type(t)))
            self._emit(f"{decl}{device_name(target.id)} = {code};")
        elif isinstance(target, ast.Subscript):
            # A string subscript on a structure value writes a member: `s["i"] = 1` becomes
            # `s.i = 1`. Recognized before the array path, which rejects a structure base
            # because a structure is a value, not a pointer.
            member:Optional[Translated] = self._struct_member_access(target)
            if member is not None:
                # A member store makes the structure parameter writable: it must not be
                # declared `const` (see render_kernel / render_device).
                self._record_member_write(member[0])
                # The generated field assignment goes through a parenthesised member
                # expression, which the static read/write analysis does not attribute to the
                # array; record the target so the host still reads the array back.
                if isinstance(target.value, ast.Subscript):
                    try:
                        self.atomic_targets.add(self._array_name(target.value.value))
                    except SyntaxError:
                        pass
                self._emit(f"{member[0]} = {code};")
                return
            # A store into a work-group buffer flattens like the read side, and has to be
            # recognized before the array path (a slice is not a pointer parameter).
            slice_code, slice_t = self._visit_expression(target.value)
            if isinstance(slice_t, tuple) and slice_t and slice_t[0] in WORK_GROUP_VALUES:
                store_code, store_t = self._visit_shared_index(slice_code, slice_t, target)
                store_code, _ = self._scalar_position(store_code, store_t, target)
                self._emit(f"{store_code} = {code};")
                return
            base_code, base_t = self._visit_array_target(target.value)
            if is_image(base_t):
                self._emit_image_write(base_code, base_t, target, code)
                return
            index_code, _ = self._visit_subscript_index(target)
            code = self._complex_store(base_t, t, code)
            self._emit(f"{base_code}[{index_code}] = {code};")
        elif isinstance(target, ast.Attribute):
            # A CUDA swizzle store (`c.xy = uv`) has no lvalue in CUDA C, so it is the
            # components' stores, one statement each (see _emit_vec_member_store). The
            # plain component store and the whole-structure store follow through
            # _visit_target below.
            if self._emit_vec_member_store(target, code, t, node):
                return
            # `p.x = ...`: a C member store on a structure value, the write twin of the
            # attribute read. The read/write analysis cannot attribute the field store to
            # the caller's argument (it sees a member expression, not a subscript), so this
            # path is only reached for synthesized object arguments, whose host side copies
            # the final member values home after the launch.
            target_code, target_t = self._visit_target(target)
            # A member store on a structure parameter makes it writable, exactly as the
            # string-subscript store does; without this the parameter was declared `const`
            # and the generated store failed to compile.
            self._record_member_write(target_code)
            if isinstance(target.value, ast.Subscript):
                # A record-array element store (`rows[i].i = ...`) writes the array the
                # parameter names, so the parameter is not read-only either.
                try:
                    self.member_writes.add(self._array_name(target.value.value))
                except SyntaxError:
                    pass
            if self._target_is_bool_member(target, target_t, t):
                # A bool member is a byte on the device, so a float would truncate
                # (`front_face = 0.99` -> 0). The store keeps boolean semantics instead.
                code = f"(({code}) != 0)"
            self._emit(f"{target_code} = {code};")
        else:
            self._error(f"unsupported assignment target: {target.__class__.__name__}", node)


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
