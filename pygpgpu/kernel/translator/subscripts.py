from __future__ import annotations
import ast
import re
from typing import (Any, Dict, List, Optional, Tuple, cast)
from ..structs import dtype_for as structs_dtype_for
from .notation import (
    BLOCK_ARRAY,
    BLOCK_LOCAL_BUILTINS,
    BLOCK_SLICE,
    PER_GROUP_BUILTINS,
    SHARED_SLICE,
    WORK_GROUP_SLICES,
    DeviceType,
    Translated,
)
from .receivers import (
    _receiver_param_of,
)
from .vectors import (
    _C_FIELD_TYPES,
    _vec_component_name,
    _vec_field_names,
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


class _SubscriptsMixin:

    def _visit_subscript(self, node:ast.Subscript)->Translated:
        value:ast.expr = node.value
        # `index[0]` on an ndenumerate index held as one name: python's index is a tuple
        # of coordinates, and a constant ordinal reads one of them. The head already
        # emitted the decomposition (see the ndenumerate branch of
        # _translate_magic_loop), so the subscript is that coordinate's own name.
        if isinstance(value, ast.Name) \
                and value.id in self._ndenumerate_index_names \
                and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, int):
            coordinates:List[str] = self._ndenumerate_index_names[value.id]
            ordinal:int = int(node.slice.value)
            if 0 <= ordinal < len(coordinates):
                return coordinates[ordinal], "int"
            self._error(
                f"the ndenumerate index has {len(coordinates)} coordinate(s); "
                f"[{ordinal}] is outside it", node)
        if isinstance(value, ast.Attribute) and value.attr == "shape":
            array_name:str = self._array_name(value.value)
            t:DeviceType = self._lookup(array_name)
            # A private array created in the body has a constant shape, so `.shape[d]` is a
            # compile-time number just like for a parameter (see _shape_var).
            private:bool = (isinstance(t, tuple) and len(t) == 4 and t[0] == "bodyarr")
            if not is_ptr(t) and not private:
                self._error("'.shape' is only available on array parameters", node)
            index:ast.expr = node.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, int):
                return self._shape_var(array_name, index.value), "int"
            self._error("only `<array>.shape[<constant>]` is supported", node)

        # A string subscript on a structure value reads a member: numpy's spelling for a
        # record field, emitted as C member access. It covers nested structures
        # (`s["inner"]["a"]`) because the member itself is a structure-typed value.
        member:Optional[Translated] = self._struct_member_access(node)
        if member is not None:
            return member

        # `box[i]` on a class that defines `__getitem__` is that method, exactly as
        # `box.attr` is `box.__getattribute__`. The member read above already covered the
        # string spelling (`s["field"]`), so only a computed index reaches here.
        if (not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str))
                and isinstance(node.value, ast.Name)):
            # Only a plain name is a class instance this class's `__getitem__` can answer
            # for. A shared buffer (`bins[blockIdx]`) and an array are indexed by the paths
            # below, and visiting their base here would raise the shared-buffer refusal
            # before that path could handle the outer subscript.
            binding:DeviceType = self._lookup(node.value.id)
            # A structure travels as a pointer too, so `is_ptr` alone would call it an
            # array and skip the `__getitem__` its class may define.
            is_struct_ptr:bool = (isinstance(binding, tuple) and is_ptr(binding)
                                  and structs_dtype_for(binding[1]) is not None)
            is_array:bool = (is_ptr(binding) and not is_struct_ptr) or (
                isinstance(binding, tuple) and len(binding) == 4 and binding[0] == "bodyarr")
            is_shared:bool = (isinstance(binding, tuple) and len(binding) == 5
                              and binding[0] in ("shared", BLOCK_ARRAY))
            if not is_array and not is_shared:
                base_code, base_t = self._visit_expression(node.value)
                owner:Optional[Tuple[type, Any]] = self._magic_owner(base_t, "__getitem__")
                if owner is not None:
                    _cls, method = owner
                    index_code, index_t = self._visit_expression(node.slice)
                    return self._emit_device_call(
                        "__getitem__", method,
                        [(base_code, base_t), (index_code, index_t)],
                        node, self_param=_receiver_param_of(method),
                        receiver_as_argument=True)

        # `buf[k]` on a `local_array_per_group(...)` value: the index selects a work group,
        # and only the caller's own group is reachable from inside a work item, so `k` must
        # be the block id. The result is that group's slice, which is then indexed normally.
        shared:Optional[Translated] = self._visit_shared_slice(node)
        if shared is not None:
            return shared

        # `buf[i]` on a `local_array(...)`/`shared_array(...)` value: this block's own
        # storage, so every subscript is an ordinary element index and no group selection
        # happens at all. Read here so the flattening below sees a plain buffer.
        block:Optional[Translated] = self._visit_block_array(node)
        if block is not None:
            return block

        # A slice used directly as an index (`a[:][i]`, `a[2:][j]`) is the same pointer at
        # an offset: emit the offset arithmetically so the expression needs no local. This
        # has to come *before* the generic visit below, which would try to evaluate the
        # slice as an expression of its own (and there is no Slice value on the device).
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Slice):
            return self._visit_sliced_subscript(node)

        # A work-group slice remembers the shape it was declared with, so a multidimension
        # subscript can be flattened exactly like an array parameter's.
        readonly_code, readonly_t = self._visit_expression(value)
        if isinstance(readonly_t, tuple) and readonly_t and readonly_t[0] in WORK_GROUP_SLICES:
            element_code, element_t = self._visit_shared_index(readonly_code, readonly_t, node)
            return self._scalar_position(element_code, element_t, node)

        base_code, base_t = self._visit_array_target(value)
        if is_image(base_t):
            return self._emit_image_read(base_code, base_t, node)
        index_code, _ = self._visit_subscript_index(node)
        return f"{base_code}[{index_code}]", base_t


    def _visit_sliced_subscript(self, node:ast.Subscript)->Translated:
        """`a[lo:hi][i]`: the element of the slice, as `array[lo + i]` on the buffer.

        A slice is a pointer at an offset, and indexing it is indexing that offset. Folding
        the two into one subscript is what lets the spelling appear in an expression without
        a local to hold the view. A slice with a step is refused by `_emit_array_slice`'s
        rules, and a nested slice (`a[1:][1:]`) is refused here rather than guessed at.
        """
        sliced:ast.Subscript = cast(ast.Subscript, node.value)
        base:ast.expr = sliced.value
        slice_node:ast.Slice = cast(ast.Slice, sliced.slice)
        if isinstance(base, ast.Subscript) and isinstance(base.slice, ast.Slice):
            self._error("a slice of a slice is not supported: take one slice into a local "
                        "and index it", node)
        base_code, base_t = self._visit_expression(base)
        if not is_ptr(base_t):
            self._error("only an array parameter can be sliced", node)
        step:Optional[ast.expr] = slice_node.step
        if step is not None and _constant_int_of(step) != 1:
            if _constant_int_of(step) == -1:
                self._error(
                    "a reversed slice (`a[::-1]`) is not a device pointer: a work item "
                    "cannot walk backwards through a buffer with one, so read the elements "
                    "with an explicit index instead (`a[n - 1 - i]`)", step)
            self._error("a slice step must be the constant 1", step)
        start:str = "0" if slice_node.lower is None else self._visit_expression(slice_node.lower)[0]
        index_code, index_t = self._visit_expression(node.slice)
        if not is_int(index_t):
            self._error("array index must be an integer", node)
        # `_visit_expression` answered the *pointer* type (a subscript on a name is a
        # pointer), so the element type is what the caller expects back.
        return f"{base_code}[({start}) + ({index_code})]", (cast(tuple, base_t))[1]


    def _visit_shared_index(self, code:str, t:object, node:ast.Subscript)->Translated:
        """Index a work-group buffer: `buf[blockIdx][i]`, `[i, j]` and `[i][j]` all flatten.

        Both kinds of work-group storage reach here, and they differ only in what the
        marker's shape starts from: a per-group buffer's value *is* the caller's slice once
        the group subscript has folded away, while a block-local buffer never had one. So
        the subscript a body writes is the same row-major flattening either way.

        A tuple subscript consumes several axes at once; a single subscript consumes one and
        hands back a slice with the remaining axes, so a nested spelling reaches the same
        element (`buf[blockIdx][row][col]` is `buf[row * 4 + col]`).

        A ``block-local`` buffer arrives as its *binding* when the store path visits the base
        name directly (`bins[i] = v`), and as its slice once the group subscript has folded;
        both are normalized to the slice form here, which is the one shape the flattening
        below reads.
        """
        if isinstance(t, tuple) and t and t[0] == BLOCK_ARRAY:
            t = (BLOCK_SLICE, t[1], tuple(t[4]))
        marker:str = t[0]
        ctype:object = t[1]
        shape:Tuple[int, ...] = tuple(t[2])
        offset_base:int = t[3] if len(t) > 3 else 0
        index:ast.expr = node.slice
        if isinstance(index, ast.Tuple):
            elements:List[ast.expr] = list(index.elts)
        else:
            elements = [index]
        if len(elements) > len(shape):
            self._error(
                f"this {self._work_group_spelling(t)} buffer has {len(shape)} dimension(s) "
                f"({', '.join(str(e) for e in shape)}), so it cannot be indexed with "
                f"{len(elements)}", index)
        terms:List[str] = []
        for position, element in enumerate(elements):
            element_code, element_t = self._visit_expression(element)
            if not is_int(element_t):
                self._error(f"an index into a {self._work_group_spelling(t)} buffer must be an "
                            f"integer", element)
            # A negative index counts from that axis's extent, exactly as it does on an
            # array parameter. Left raw it addressed the shared array's own header.
            term:str = self._python_index(element_code, element,
                                          literal_extent=shape[position])
            for extent in shape[position + 1:len(shape)]:
                term = f"({term}) * {extent}"
            terms.append(term)
        flat:str = " + ".join(terms) if terms else "0"
        remaining:Tuple[int, ...] = shape[len(elements):]
        if not remaining:
            if offset_base:
                return f"{code}[({offset_base}) + ({flat})]", ctype
            return f"{code}[{flat}]", ctype
        # Partly indexed: the caller's code keeps the running offset and the axes left. Such a
        # value is a *slice*, not an element, so it can only be indexed further -- writing to
        # it would target the whole buffer. `_scalar_position` refuses that where a scalar is
        # required.
        combined:str = f"({offset_base}) + ({flat})" if offset_base else flat
        return f"{code}", (marker, ctype, remaining, combined)


    def _scalar_position(self, code:str, t:object, node:ast.AST)->Translated:
        """A work-group buffer's element, refusing a partly indexed one.

        `buf[blockIdx]` with `(2, 2)` axes and one index is a slice of a row; it has no
        device value of its own, so using it where a scalar is needed is an error rather than
        a buffer with no subscript.
        """
        if isinstance(t, tuple) and t and t[0] in WORK_GROUP_SLICES:
            spelling:str = self._work_group_spelling(t)
            hint:str = ("`buf[blockIdx][i, j]`" if t[0] == SHARED_SLICE else "`buf[i, j]`")
            self._error(
                f"this {spelling}() buffer has {len(t[2])} dimension(s) left, so it is a slice "
                f"rather than an element: index every axis, as {hint}", node)
        return code, t


    def _visit_block_array(self, node:ast.Subscript)->Optional[Translated]:
        """`buf[i]` on a `local_array(...)`/`shared_array(...)` value: this block's storage.

        There is nothing to select: the value already *is* the caller's block, so the
        subscript is an ordinary element index and the flattening of the following axes is
        the same row-major one an array parameter gets. The marker carries the declared
        shape, so the axes are known at translation time.
        """
        if not isinstance(node.value, ast.Name):
            return None
        t:DeviceType = self._lookup(node.value.id)
        if not (isinstance(t, tuple) and len(t) == 5 and t[0] == BLOCK_ARRAY):
            return None
        ctype, c_name, shape = t[1], t[2], t[4]
        return self._visit_shared_index(c_name, (BLOCK_SLICE, ctype, shape), node)


    def _work_group_spelling(self, t:object)->str:
        """The builtin name a work-group buffer's error message should use.

        Both pairs build one declaration, but an error has to name the spelling the body
        actually wrote, and the marker's first slot is what says which of the two it came
        from -- including on a partly indexed slice, which keeps its origin.
        """
        return (PER_GROUP_BUILTINS[0] if isinstance(t, tuple) and t and t[0] in (
            SHARED_SLICE, BLOCK_ARRAY) else BLOCK_LOCAL_BUILTINS[0])


    def _visit_shared_slice(self, node:ast.Subscript)->Optional[Translated]:
        """`buf[blockIdx]` on a per-group buffer: **this** group's storage, and nothing more.

        The subscript picks *which* group's slice is meant, which is a real choice on the
        host (where the value is a mapping) but not on a device: the hardware already
        gives every group its own `__shared__`/`__local` storage, so the answer is always
        the caller's own slice and **no index arithmetic is emitted at all**. `buf[blockIdx]`
        folds to the buffer itself.

        The subscript is therefore validated rather than computed: it must be a block id
        (`blockIdx`, or `get_group_id(d)`), because any other expression would name a group
        a work item cannot reach. Emitting that as an offset would be worse than refusing --
        it would read the next group's copy, which is precisely the memory a work item must
        never touch.
        """
        if not isinstance(node.value, ast.Name):
            # The inner `buf[block_id(0)]` is already a pointer by the time the outer
            # `[j]` is visited, so that subscript indexes an array and is not a group
            # selection.
            return None
        t:DeviceType = self._lookup(node.value.id)
        if not (isinstance(t, tuple) and len(t) == 5 and t[0] == "shared"):
            return None
        ctype, c_name, shape = t[1], t[2], t[4]
        index:ast.expr = node.slice
        if not self._is_block_id_expression(index):
            self._error(
                "a local_array_per_group() buffer is one slice per work group, and a work "
                "item can only reach its own: index it with the group first, as "
                "`buf[blockIdx][j]` (or `buf[get_group_id(0)][j]`). A buffer this block "
                "indexes directly is `local_array(...)`", index)
        # Both backends give each work group its own copy of the buffer -- OpenCL allocates
        # one `__local` parameter per group, CUDA/HIP a `__shared__` array per block -- so
        # the slice a work item reaches is the whole buffer, and the block id only names
        # which group is asking. No offset is added: doing so would read into the next
        # group's copy, which is exactly the memory a work item must not touch.
        return c_name, (SHARED_SLICE, ctype, shape)


    def _is_block_id_expression(self, node:ast.expr)->bool:
        """Whether an expression names a work group, i.e. evaluates to the block id.

        Both spellings qualify -- `block_id(d)` and `get_group_id(d)` -- and so does a local
        the body assigned from one (`gid = block_id(0)`, then `buf[gid]`), because it is the
        same number by another name.
        """
        if isinstance(node, ast.Name):
            # `blockIdx` is the group vector itself; a local holding one is the same axis
            # under another name.
            return node.id == "blockIdx" or self._lookup(node.id) == "blockid"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            # `blockIdx.x`: one axis of the group vector.
            return node.value.id == "blockIdx" and node.attr in ("x", "y", "z")
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return False
        return self._builtin_call_name(node) == "get_group_id"


    def _struct_member_access(self, node:ast.Subscript)->Optional[Translated]:
        """`s["field"]` on a structure value, or `v["x"]` on a vector, or None.

        A string subscript is numpy's spelling for a record field, and it works for a vector
        too (`v["x"]`). Both are member access in C, so both are emitted the same way.
        """
        from .. import structs

        index:ast.expr = node.slice
        if not (isinstance(index, ast.Constant) and isinstance(index.value, str)):
            return None
        base_code, base_t = self._visit_expression(node.value)
        field:str = index.value

        if is_vec(base_t):
            component:str = _vec_component_name(field)
            if component not in _vec_field_names(base_t):
                self._error(
                    f"vector type {base_t} has no component '{field}' "
                    f"(available: {''.join(_vec_field_names(base_t))})", node)
            return f"({base_code}).{component}", _C_FIELD_TYPES[(base_t, component)]

        dtype:Optional[object] = structs.dtype_for(base_t) if isinstance(base_t, str) else None
        if dtype is None and (isinstance(base_t, tuple) and is_ptr(base_t)):
            dtype = structs.dtype_for(base_t[1])
        if dtype is None:
            return None
        members:Dict[str, object] = structs.member_types(dtype, self.backend)
        if field not in members:
            self._error(
                f"structure value has no member '{field}' "
                f"(available: {', '.join(members)})", node)
        # The parenthesised member expression is what C expects; a pointer (a structure
        # argument always travels as one) goes through '->', a value through '.'. A record
        # array's element (`g[0]`) is a value, so its own store form is `.`, while the
        # parameter itself is the pointer.
        if isinstance(base_t, tuple) and is_ptr(base_t) and not base_code.startswith("&"):
            if re.search(r"\]\s*$", base_code):
                return f"({base_code}).{field}", members[field]
            return f"({base_code})->{field}", members[field]
        return f"({base_code}).{field}", members[field]
