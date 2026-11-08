from __future__ import annotations
import ast
from typing import (Dict, List, Optional, Tuple)
import numpy as np
from ..structs import dtype_for as structs_dtype_for, member_types as structs_member_types
from .notation import (
    DeviceType,
    OuterVarEntry,
    SHARED_SLICE,
    Translated,
)
from .vectors import (
    is_vec,
)
from .predicates import (
    is_image,
    is_int,
    is_ptr,
)
from .walk import (
    _constant_int_of,
)


class _TargetsMixin:

    def _visit_target(self, target:ast.expr)->Translated:
        if isinstance(target, ast.Name):
            outer:Optional[OuterVarEntry] = self.outer_vars.get(target.id)
            if outer is not None:
                if not outer["writable"]:
                    self._error(
                        f"cannot assign to the outer variable '{target.id}': without a "
                        f"'nonlocal {target.id}' or 'global {target.id}' declaration this "
                        f"would create a local variable, shadowing the outer one", target)
                # A store through the parameter as `name[0]` rather than `(*name)`: both are
                # the same location, but only the subscript form is recognized as a write
                # by the static read/write analysis, which decides whether the value is
                # read back to the host.
                return f"{target.id}[0]", outer["type"]
            t:DeviceType = self._lookup(target.id)
            if t is None:
                self._error(f"variable '{target.id}' used before assignment", target)
            if is_ptr(t):
                self._error(f"cannot assign to array parameter '{target.id}'", target)
            return target.id, t
        if isinstance(target, ast.Subscript):
            # A string subscript on a structure value writes a member: `s["i"] = 1` becomes
            # `s.i = 1`. It has to be recognized before the array path, which would reject a
            # structure base (it is a value, not a pointer).
            member:Optional[Translated] = self._struct_member_access(target)
            if member is not None:
                return member
            # A store into a zeros_per_group() slice: the same flattening as the read side.
            slice_code, slice_t = self._visit_expression(target.value)
            if isinstance(slice_t, tuple) and slice_t and slice_t[0] == SHARED_SLICE:
                element_code, element_t = self._visit_shared_index(slice_code, slice_t, target)
                return self._scalar_position(element_code, element_t, target)
            base_code, base_t = self._visit_array_target(target.value)
            index_code, _ = self._visit_subscript_index(target)
            self._record_element_write(target.value)
            return f"{base_code}[{index_code}]", base_t
        if isinstance(target, ast.Attribute):
            # `p.x = ...` on a structure value: a C member store. The read side already
            # handles attribute access on structures; this is its write twin. A written
            # object argument travels as a one-element record array, so the store goes
            # through the element (`p[0].x = ...`), which is what makes it reach the host.
            code:str
            t:object
            code, t = self._visit_expression(target.value)
            if isinstance(t, tuple) and is_ptr(t):
                dtype:object = structs_dtype_for(t[1])
                if dtype is None:
                    self._error(
                        f"cannot assign to '{ast.unparse(target)}': {t[1]} is not a "
                        f"structure the translator knows", target)
                # `box.total = v` is a call to the property's setter, next to the getter
                # the read side already answers.
                member_types:Dict[str, object] = structs_member_types(dtype, self.backend)
                if target.attr not in member_types:
                    self._error(
                        f"structure value has no member '{target.attr}' "
                        f"(available: {', '.join(member_types)})", target)
                # The static analysis sees a member expression, not a subscript, so it never
                # marks the array as written; without this flag the host would not read the
                # stores back. `atomic_targets` is exactly the force-writable channel.
                self.atomic_targets.add(target.value.id)
                if code.startswith("&"):
                    # A method's own receiver, by address: a plain pointer store.
                    return f"({code})->{target.attr}", member_types[target.attr]
                return f"({code}[0]).{target.attr}", member_types[target.attr]
            if is_vec(t) or not isinstance(t, str) or structs_dtype_for(t) is None:
                self._error(
                    f"cannot assign to '{ast.unparse(target)}': member stores need a "
                    f"structure value, and {t} is not one", target)
            member_types:Dict[str, object] = structs_member_types(structs_dtype_for(t), self.backend)
            if target.attr not in member_types:
                self._error(
                    f"structure value has no member '{target.attr}' "
                    f"(available: {', '.join(member_types)})", target)
            return f"({code}).{target.attr}", member_types[target.attr]
        self._error(f"unsupported assignment target: {target.__class__.__name__}", target)


    def _target_is_bool_member(self, target:ast.expr, target_t:object, value_t:object)->bool:
        """Whether a member store lands in a bool-typed field while holding a wider type."""
        from .. import structs  # noqa: PLC0415

        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)):
            return False
        if value_t == "bool":
            return False
        base_t:DeviceType = self._lookup(target.value.id)
        struct_t:object = base_t[1] if is_ptr(base_t) else base_t
        dtype:object = structs.dtype_for(struct_t) if isinstance(struct_t, str) else None
        if dtype is None or target.attr not in (dtype.names or ()):
            return False
        return np.dtype(dtype.fields[target.attr][0]).name == "bool"


    def _visit_array_target(self, node:ast.expr)->Translated:
        code, t = self._visit_expression(node)
        if is_image(t):
            # An image keeps its own marker: indexing it means sampling, not
            # pointer arithmetic, so the caller decides from the marker.
            return code, t
        if not is_ptr(t):
            self._error("only array (numpy) and image arguments can be indexed", node)
        return code, t[1]


    def _record_subscript_indices(self, node:ast.Subscript)->None:
        """Remember how many indices each array is subscripted with.

        A pointer parameter carries no rank, so the translator cannot tell whether
        ``out[i, j]`` addresses one element or selects a slice; the value passed at call
        time can. Recording the count here lets the launch compare it with the array's
        real shape instead of silently writing one element of a 3-D array (which is what
        ``out[i, j] = v`` used to compile into).
        """
        base:ast.expr = node.value
        name:str = ""
        if isinstance(base, ast.Name):
            name = base.id
        elif (self.self_param and isinstance(base, ast.Attribute)
              and isinstance(base.value, ast.Name) and base.value.id == self.self_param):
            name = f"_self_{base.attr}"
        else:
            return
        if not is_ptr(self._lookup(name)):
            return
        count:int = len(node.slice.elts) if isinstance(node.slice, ast.Tuple) else 1
        previous:int = self.subscript_indices.get(name, 0)
        if count > previous:
            # The widest subscript is the one that decides; among equally wide ones the
            # FIRST is the one python would raise at, because it is the first the body
            # reaches. Its line is in the parsed copy's numbering, which _body_frame
            # converts back to the user's file, and the column is where the subscript
            # expression starts, so the caret underlines the indexing rather than the
            # whole statement.
            self.subscript_lines[name] = getattr(node, "lineno", 0) or 0
            self.subscript_columns[name] = getattr(node, "col_offset", 0) or 0
            self.subscript_texts[name] = ast.get_source_segment(self.source_text, node) or ""
        self.subscript_indices[name] = max(previous, count)


    def _visit_subscript_index(self, node:ast.Subscript)->Translated:
        self._record_subscript_indices(node)
        return self._flatten_index(node.slice, node.value, node)


    def _flatten_index(self, index:ast.expr, target:ast.expr,
                       node:ast.AST)->Translated:
        """One flat offset for an index that may name several axes.

        A tuple index (`a[i, j]`) is row-major flattening over the *declared* shape, which is
        why it needs the array's name; a scalar index is already a flat offset and is left
        alone, so the 1-D spelling keeps working on a multidimensional array.

        A **negative index** is python's from-the-end spelling, and emitting it raw was a
        silent out-of-bounds access: `a[-1]` read the word *before* the buffer and answered
        0.0 where python answers the last element, and `a[-1] = v` wrote outside the array.
        A constant negative index becomes `extent - n` at translation time; a computed one
        gets the same correction around a `%`-free select, so `i - n` cannot underflow into
        a neighbouring array. An extent of zero is python's own `IndexError` territory, so
        the expression is clamped to 0 rather than left negative (the caller's bounds guard
        already skips the work item when the array is empty).

        An atomic call passes the whole array as the target, so an atomic on a shared buffer
        must name the group's slice (`buf[blockIdx]`) rather than the buffer.
        """
        if isinstance(index, ast.Tuple):
            slice_shape:Optional[Tuple[int, ...]] = self._shared_slice_shape(target)
            if slice_shape is not None:
                # An atomic on a shared buffer names the group's slice as its array, and the
                # slice carries its own shape, so the flattening needs no `_shape_var`.
                dims:List[Tuple[str, object]] = [
                    (self._visit_expression(e), e) for e in index.elts]
                terms:List[str] = []
                for d, ((code, t), element) in enumerate(dims):
                    if not is_int(t):
                        self._error("array indices must be integers", node)
                    code = self._python_index(
                        code, element,
                        literal_extent=(slice_shape[d] if d < len(slice_shape) else None))
                    term:str = code
                    for inner in range(d + 1, len(dims)):
                        if inner >= len(slice_shape):
                            self._error(
                                f"this shared buffer has {len(slice_shape)} dimension(s), so "
                                f"it cannot be indexed with {len(dims)}", node)
                        term = f"{term} * {slice_shape[inner]}"
                    terms.append(term)
                return " + ".join(terms), "int"
            if self._lookup(self._array_name(target)) is not None \
                    and isinstance(self._lookup(self._array_name(target)), tuple) \
                    and self._lookup(self._array_name(target))[0] == "shared":
                self._error(
                    "a shared buffer is indexed by its group first, so pass the tuple after "
                    "the group: `buf[blockIdx][i, j]`", node)
            array_name:str = self._array_name(target)
            dims = [(self._visit_expression(e), e) for e in index.elts]
            terms = []
            for d, ((code, t), element) in enumerate(dims):
                if not is_int(t):
                    self._error("array indices must be integers", node)
                code = self._python_index(code, element, target, dim=d)
                term = code
                for inner in range(d + 1, len(dims)):
                    term = f"{term} * {self._shape_var(array_name, inner)}"
                terms.append(term)
            return " + ".join(terms), "int"
        code, t = self._visit_expression(index)
        if not is_int(t):
            self._error("array index must be an integer", node)
        return self._python_index(code, index, target), t


    def _python_index(self, code:str, node:ast.expr,
                      target:Optional[ast.expr]=None, dim:int=0,
                      literal_extent:Optional[int]=None)->str:
        """A python index as the device reads it, with a constant negative index folded.

        `a[-1]` is python's last element, and the device has no such addressing: the index
        is a plain offset, so a raw `-1` reaches memory *before* the buffer. A constant
        negative index is therefore rewritten as `extent - n`, which is the element python
        names -- `a[-1]` and `a[n - 1]` are the same access, so the fold costs nothing.

        The extent is looked up **only** when the fold actually happens, and that matters
        beyond tidiness: `_shape_var` is what *registers* an extent parameter, so asking
        for one eagerly made every kernel subscript declare a shape it never read. A
        gid-style Metal body then grew an `_out_shape0` parameter it is defined not to
        have, and a 2-D one grew axis 0 as well as axis 1.

        A *computed* index is passed through unchanged, deliberately. Wrapping it in a
        sign test would evaluate the expression twice (wrong for a call or a subscript) and
        would put a select around every array access in every kernel to pay for a case that
        is an out-of-range index by python's own rules -- `a[k]` with a negative `k` is the
        same kind of mistake as `a[100]` on a short array, which the device has never
        bounds-checked either.
        """
        constant:Optional[int] = _constant_int_of(node)
        if constant is None or constant >= 0:
            return code
        extent:Optional[str] = None
        if literal_extent is not None:
            extent = str(literal_extent)
        elif target is not None:
            try:
                extent = self._shape_var(self._array_name(target), dim)
            except SyntaxError:
                extent = None
        if extent is None:
            self._error(
                "a negative index needs the array's extent to count from the end, and "
                "the array this index addresses is not known here", node)
        # `extent - n` is python's answer. If the array is shorter than the index python
        # raises IndexError, and no work item can raise it, so the offset is clamped.
        needed:int = -constant
        return f"((({extent}) - {needed} >= 0) ? (({extent}) - {needed}) : 0)"
        if constant is not None:
            if constant >= 0:
                return code
            if extent is None:
                self._error(
                    "a negative index needs the array's extent to count from the end, and "
                    "the array this index addresses is not known here", node)
            # `extent - n` is python's answer. If the array is shorter than the index python
            # raises IndexError, and no work item can raise it, so the offset is clamped.
            needed:int = -constant
            return f"((({extent}) - {needed} >= 0) ? (({extent}) - {needed}) : 0)"
        if extent is None:
            return f"((({code}) >= 0) ? ({code}) : 0)"
        return f"((({code}) >= 0) ? ({code}) : (({code}) + ({extent})))"


    def _shared_slice_shape(self, node:ast.expr)->Optional[Tuple[int, ...]]:
        """The shape a `zeros_per_group` slice was declared with, or None if it is not one."""
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            return None
        t:DeviceType = self._lookup(node.value.id)
        if not (isinstance(t, tuple) and len(t) == 5 and t[0] == "shared"):
            return None
        if not self._is_block_id_expression(node.slice):
            return None
        shape:Tuple[int, ...] = tuple(t[4])
        return shape
