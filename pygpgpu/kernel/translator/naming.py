from __future__ import annotations
import ast
from typing import (List, Optional, Set, Tuple, cast)
from .notation import (
    DeviceType,
    device_name,
)
from .predicates import (
    is_image,
    is_ptr,
)
from .walk import (
    _constant_int_of,
    _declared_scope_names,
)


class _NamingMixin:

    def _emit_array_slice(self, name:str, node:ast.Assign)->bool:
        """Emit `local = array[lo:hi]` as a pointer into the buffer, and record its length.

        Python's slice is a *view*: it shares the buffer and adds an offset and a length.
        The device has no view type, so the same thing is spelled with a pointer it already
        has -- `T* part = a + lo;` -- and the length is kept in `_array_slice_extents`, which
        is what `part.shape[0]` reads. `a[n // 2:]` works the same way: the bound is an
        expression, and the emitted pointer just carries it.

        Two facts make the pointer arithmetic compose. The offset is added to the **base's
        own translated code**, so a slice of a slice (`inner = part[1:3]`) starts from
        `part`, not from the array the chain began with. And an omitted upper bound is the
        **immediate** base's length, so `part[1:]` is `len(part) - 1` rather than the whole
        array's extent minus one.

        A step other than 1 needs a real view (a work item cannot walk a buffer backwards),
        and a *rank* above one is refused at the call, where the shape is known: on a 2-D
        array `grid[0:2]` is two rows, which one offset cannot express.
        """
        subscript:ast.Subscript = cast(ast.Subscript, node.value)
        slice_node:ast.Slice = cast(ast.Slice, subscript.slice)
        base:ast.expr = subscript.value
        immediate:str = self._array_name(base)
        root:str = self._array_pointer_root(immediate)
        base_code, base_t = self._visit_expression(base)
        if not is_ptr(base_t):
            return False
        step:Optional[ast.expr] = slice_node.step
        if step is not None and _constant_int_of(step) != 1:
            if _constant_int_of(step) == -1:
                self._error(
                    "a reversed slice (`a[::-1]`) is not a device pointer: a work item "
                    "cannot walk backwards through a buffer with one, so read the elements "
                    "with an explicit index instead (`a[n - 1 - i]`)", step)
            self._error("a slice step must be the constant 1", step)
        lower:Optional[ast.expr] = slice_node.lower
        upper:Optional[ast.expr] = slice_node.upper
        start:str = "0" if lower is None else self._visit_expression(lower)[0]
        # An omitted upper bound is the length of what is being sliced -- the array's extent
        # for a parameter, the parent slice's own length for a slice of a slice.
        stop:str = (self._shape_var(immediate, 0) if upper is None
                    else self._visit_expression(upper)[0])
        self._define(name, base_t, node)
        self._emit(f"{self._device_type(base_t[1])}* {device_name(name)} = "
                   f"{base_code} + ({start});")
        # Two different questions are answered by two tables here, which is why both are
        # written. (1) The *length* is the slice's own, so `part.shape[0]` reads it -- that
        # is the loop's bounds guard. (2) The *launch size* has to come from a parameter,
        # and a slice is not one, so the sizing resolves to the array the chain began with:
        # the grid then covers more work items than the slice has, and the guard above skips
        # the excess. That is exactly how a rounded-up automatic work size already works.
        self._array_slice_extents[name] = f"(({stop}) - ({start}))"
        self._array_pointer_sources[name] = root
        if root not in self.sliced_arrays:
            self.sliced_arrays.append(root)
        return True


    def _loop_domain_name(self, attr_node:ast.expr)->Optional[str]:
        """The name `X` in `X.shape[0]`, exactly as the body wrote it.

        No alias or slice resolution happens here: the caller needs to know *which local*
        the bound was read from, because a slice's extent is its own length while its
        launch size is the array's.
        """
        if not (isinstance(attr_node, ast.Attribute) and attr_node.attr == "shape"):
            return None
        base:ast.expr = attr_node.value
        if isinstance(base, ast.Name):
            return base.id
        if (self.self_param and isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name) and base.value.id == self.self_param):
            return f"_self_{base.attr}"
        return None


    def _pointer_source_name(self, value:ast.expr)->Optional[str]:
        """The array a pointer assignment copies from, when the right-hand side is one.

        Only a bare name (`src = a`, resolved through the alias chain) or a receiver member
        (`buf = self.data`) counts. Every other attribute is a *value* -- `lane =
        threadIdx.x` is the usual one -- and asking `_array_name` about it would refuse a
        statement that has nothing to do with arrays.
        """
        if isinstance(value, ast.Name):
            return self._array_pointer_root(value.id)
        if (self.self_param and isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name) and value.value.id == self.self_param):
            return f"_self_{value.attr}"
        return None


    def _record_element_write(self, base:ast.expr)->None:
        """A store into an element makes the *array it came from* writable.

        The read/write analysis reads the generated source, where a local pointer
        (`double* part = a + 1;`) is not the parameter -- so a store through the local would
        leave `a` marked read-only and the host would drop the write silently. The name is
        followed to the array the local points at, and that name is the force-writable
        channel the member-store path already uses (see `atomic_targets`).
        """
        try:
            name:str = self._array_name(base)
        except SyntaxError:
            return
        root:str = self._array_pointer_root(name)
        if is_ptr(self._lookup(root)):
            self.atomic_targets.add(root)


    def _record_member_write(self, member_code:str)->None:
        """Remember that this structure parameter's members are written.

        The generated expression is `(base)->field` or `(base[0]).field`; the base is the
        parameter name, and that parameter must not be declared `const`.
        """
        import re as _re

        # The base is the parameter name, optionally wrapped and optionally indexed
        # (`(g[0]).field` is a record-array element of the parameter `g`).
        match = _re.search(r"(?:\*?&&?)?\(?([A-Za-z_]\w*)", member_code)
        base:str = match.group(1) if match else ""
        if base:
            self.member_writes.add(base)
            # The alias case (`q = p; q.a = ...`) writes the same buffer.
            alias:Optional[DeviceType] = (self.scopes[-1].get(base)
                                          if self.scopes else None)
            if isinstance(alias, tuple) and len(alias) == 2 and alias[0] == "alias":
                self.member_writes.add(alias[1])


    def _shape_var(self, array_name:str, dim:int)->str:
        # A local pointer (`src = a`) is a *device variable*, but it has no extent parameter
        # of its own: the extent lives on the array it was assigned from, because that is
        # what the launch passes. Resolving here is what makes `src.shape[0]` compile -- and
        # it is the single place that decides it, because every extent parameter is built by
        # this call.
        # A slice (`part = a[lo:hi]`) has a length of its own, recorded when it was bound.
        if dim == 0 and array_name in self._array_slice_extents:
            return f"({self._array_slice_extents[array_name]})"
        array_name = self._array_pointer_root(array_name)
        # A private array created in the work-item body has a constant shape, so its extents
        # are compile-time numbers rather than launch parameters.
        binding:DeviceType = self._lookup(array_name)
        if isinstance(binding, tuple) and len(binding) == 4 and binding[0] == "bodyarr":
            shape:Tuple[int, ...] = binding[2]
            if 0 <= dim < len(shape):
                return str(shape[dim])
        key = (array_name, dim)
        if key not in self.shapes_used:
            self.shapes_used.append(key)
        # The extent parameter is named after the array it bounds, and an array called
        # `register` or `new` would put the reserved word inside the identifier. Mapping the
        # array name here is the one place that decides the spelling, because every extent
        # parameter is built by this call.
        return f"_{device_name(array_name)}_shape{dim}"

    def _tail_var(self, array_name:str, count:int)->str:
        """The launch parameter holding a sub-array's extent after a partial index.

        A tuple index naming fewer axes than the array has selects a sub-array in python;
        its elements are the ``prod(shape[count:])`` contiguous words that follow the
        named flattening. The rank is a launch fact a pointer cannot carry, so the product
        travels as one parameter named after the array and the index count -- ``1`` when
        the index is full-rank. The registration rides in `shapes_used` with a negative
        dim marker, so the parameter is declared and bound exactly where the extents are.
        """
        key = (array_name, -count - 2)
        if key not in self.shapes_used:
            self.shapes_used.append(key)
        return f"_{device_name(array_name)}_tail{count}"


    def _array_pointer_root(self, name:str)->str:
        """The array a local pointer variable was assigned from, or the name itself.

        An array parameter reaches the device as `T*`, and `src = a` declares one more
        pointer (`T* src = a;`) rather than copying anything. The pointer itself has no
        extent, though: extents are launch parameters named after the *array*, so a shape
        read through the local has to resolve to the array it was assigned from. A chain
        (`x = src`) follows to the first name, bounded so a malformed pair cannot loop.

        The table is filled ahead of the walk (see `_collect_array_pointers`), because the
        *loop detection* asks about `enumerate(src)` before any statement is translated.
        """
        seen:Set[str] = set()
        while name in self._array_pointer_sources and name not in seen:
            seen.add(name)
            name = self._array_pointer_sources[name]
        return name


    def _collect_array_pointers(self, body:List[ast.stmt])->None:
        """Record every `local = <array>` in the body before it is walked.

        A pointer local is a fact about the body, and the magic-loop scan needs it before
        the statement that creates it is reached: `enumerate(src)` is only the dispatch when
        `src` is known to be an array. The sources are recorded with their own name here and
        resolved through `_array_pointer_root`, so a chain (`two = one`) works too.
        """
        for statement in body:
            if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)):
                continue
            target_name:str = statement.targets[0].id
            source:ast.expr = statement.value
            if isinstance(source, ast.Name):
                self._array_pointer_sources.setdefault(target_name, source.id)
            elif isinstance(source, ast.Attribute) and isinstance(source.value, ast.Name) \
                    and self.self_param and source.value.id == self.self_param:
                self._array_pointer_sources.setdefault(target_name, f"_self_{source.attr}")


    def _local_names(self)->Set[str]:
        """Names this function defines as locals (assigns, unpacks, loop targets).

        A name declared `global`/`nonlocal` is *not* local even though the body assigns
        to it: python binds it in the scope, which is exactly what this feature reproduces.
        An undeclared name that is both read and assigned is python's local variable (it
        shadows any outer name of the same name), so it stays local here too.
        """
        declared:Set[str] = _declared_scope_names(self.tree)
        names:Set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id not in declared:
                    names.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
        return names


    def _referenced_names(self)->Set[str]:
        """Names the body loads as a plain variable, not as an attribute or call target."""
        names:Set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
        return names


    def _is_launch_wide_name(self, name:str)->bool:
        """Whether a name's value is the same in every work item.

        The magic loop's own indices, the kernel's scalar parameters and its outer-scope
        values all are; an array element, a per-item local and a value computed inside the
        loop are not.
        """
        if name in self.work_item_indices:
            return True
        binding:DeviceType = self._lookup(name)
        if binding is None:
            # An unresolved name is a module global or a builtin -- the same for everyone.
            return True
        if isinstance(binding, tuple):
            return False
        if is_ptr(binding) or is_image(binding):
            # A pointer says nothing by itself: `x[i]` is per-item even though `x` is not.
            return False
        return True


    def _array_name(self, node:ast.expr)->str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            return self._array_name(node.value)
        if (self.self_param and isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == self.self_param):
            return f"_self_{node.attr}"
        self._error("unsupported array expression", node)
