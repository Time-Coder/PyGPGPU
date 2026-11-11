from __future__ import annotations
import ast
from typing import (Any, Dict, List, Optional, Tuple)
from ..device_builtins import (
    emit_atomic,
    float_atomic_add_source,
)
from .notation import (
    BLOCK_ARRAY,
    BLOCK_SLICE,
    WORK_GROUP_VALUES,
    Translated,
)
from .receivers import (
    MAGIC_INPLACE,
    MAGIC_METHODS,
    _receiver_param_of,
)
from .predicates import (
    is_float,
    is_image,
    is_int,
    is_ptr,
    promote,
)


class _AugassignMixin:

    def _augassign_is_race_free(self, target:ast.Subscript)->bool:
        """True when each work item provably touches a different element.

        The only case the translator can prove without knowing the launch geometry is the
        one where the subscript is built from the magic loop's own index variables: every
        index is one of them, each appears once, and all of them appear. Distinct work
        items then have distinct index tuples, so `out[i, j] += v` cannot overlap with
        another work item and a plain read-modify-write is exact.

        Anything else -- a subscript driven by a data value (`bins[image[i]] += 1`), a
        partial tuple (`out[i] += v` in a 2-D loop), or an arithmetic index -- may collide
        and is treated as racy.
        """
        indices:ast.expr = target.slice
        elements:List[ast.expr] = list(indices.elts) if isinstance(indices, ast.Tuple) else [indices]
        if not all(isinstance(element, ast.Name) for element in elements):
            return False
        names:List[str] = [element.id for element in elements]  # type: ignore[union-attr]
        work_items:List[str] = self.work_item_indices
        if not work_items:
            # A gid-style kernel states its work item through gpu.get_global_id, which this
            # check cannot relate to a subscript, so nothing is provable there.
            return False
        return sorted(names) == sorted(work_items)


    def _visit_atomic_augassign(self, node:ast.AugAssign, op_name:str,
                                atomic_name:str,
                                pointer_code:Optional[str]=None)->None:
        """Emit a racy `arr[i] op= value` as the matching device atomic operation.

        ``pointer_code`` is set when the target is an outer scalar: the device parameter is
        already a pointer to one element, so there is no subscript to compute.
        """
        if pointer_code is not None:
            name:str = node.target.id  # type: ignore[union-attr]
            outer:Dict[str, object] = self.outer_vars[name]
            self._emit_atomic_update(node, op_name, atomic_name, outer["type"],
                                     array_code=name, index_code="0",
                                     pointer_code=pointer_code)
            return

        target:ast.Subscript = node.target  # type: ignore[assignment]
        # `bins[blockIdx][value] += 1` on per-group storage, or `bins[value] += 1` on this
        # block's own storage: the buffer is the array, and any offset already consumed
        # folds into the index.
        slice_code, slice_t = self._visit_expression(target.value)
        if isinstance(slice_t, tuple) and slice_t and slice_t[0] == BLOCK_ARRAY:
            # The binding itself, not a slice: this block's array is already the whole
            # storage, so there is no group subscript and no offset to carry.
            slice_t = (BLOCK_SLICE, slice_t[1], tuple(slice_t[4]))
        if isinstance(slice_t, tuple) and slice_t and slice_t[0] in WORK_GROUP_VALUES:
            pending:str = slice_t[3] if len(slice_t) > 3 else ""
            index_code, _ = self._visit_shared_index("", slice_t, target)
            base_code:str = slice_code
            element_t:object = slice_t[1]
            flat:int = index_code.find("[")
            inner:str = index_code[flat + 1:-1] if flat >= 0 else "0"
            if pending:
                inner = f"({pending}) + ({inner})"
            self._emit_atomic_update(node, op_name, atomic_name, element_t,
                                     array_code=base_code, index_code=inner,
                                     is_local=True)
            return
        # _visit_array_target already narrows a pointer to its element type.
        base_code, element_t = self._visit_array_target(target.value)
        self._record_element_write(target.value)
        if is_image(element_t):
            self._error(
                "'op=' on an image element is not supported; write `img[x, y] = value` "
                "or use gpu.atomic_* on an array buffer", node)
        index_code, _ = self._visit_subscript_index(target)
        self._emit_atomic_update(node, op_name, atomic_name, element_t,
                                 array_code=base_code, index_code=index_code,
                                 is_local=self._is_local_buffer(target.value))


    def _emit_atomic_update(self, node:ast.AugAssign, op_name:str, atomic_name:str,
                            element_t:object, array_code:str, index_code:str,
                            pointer_code:Optional[str]=None,
                            is_local:bool=False)->None:
        """The shared tail of an atomic `op=` on an array element or an outer scalar."""
        # A float slot is addable through the portable CAS helper (see
        # float_atomic_add_source); every other non-integer element has no atomic at all.
        float_add:bool = (element_t == "float" and op_name == "Add")
        if not is_int(element_t) and not float_add:
            self._error(
                f"'{self._augassign_symbol(op_name)}=' on this location may be written by "
                f"several work items at once, and a {element_t} value has no device atomic "
                f"operation. Python would add every contribution, so the plain "
                f"read-modify-write would lose some. Use an integer counter, or accumulate "
                f"per work item and reduce afterwards (see examples/09_reduction.py)", node)
        value_code, value_t = self._visit_expression(node.value)
        if not is_int(value_t) and not is_float(value_t):
            self._error(f"'{op_name}' needs a numeric value", node)
        if pointer_code is None:
            self.atomic_targets.add(self._array_name(node.target.value))  # type: ignore[union-attr]
        else:
            self.atomic_targets.add(node.target.id)  # type: ignore[union-attr]
        if float_add and float_atomic_add_source(self.backend) \
                not in self.device_helpers:
            self.device_helpers.append(float_atomic_add_source(self.backend))
        call:str = emit_atomic(atomic_name, array_code, index_code, value_code,
                               element_t, self.backend, pointer_code=pointer_code,
                               is_local=is_local)
        self._emit(f"{call};")


    def _augassign_symbol(self, op_name:str)->str:
        return {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "FloorDiv": "//",
                "Mod": "%", "Pow": "**", "LShift": "<<", "RShift": ">>",
                "BitAnd": "&", "BitOr": "|", "BitXor": "^"}.get(op_name, op_name)


    def _visit_augassign(self, node:ast.AugAssign)->None:
        # A racy `arr[i] op= v` must not compile into a plain read-modify-write: every work
        # item reads the old value and writes it back, so contributions are lost with no
        # error at all. Python accumulates every one of them, so the device code has to use
        # an atomic operation whenever the translator cannot prove the elements are
        # distinct. `out[i] += x[i]` over the loop's own index stays a plain store.
        op_name:str = node.op.__class__.__name__
        # An outer scalar named `nonlocal`/`global` is one device location shared by every
        # work item, so an in-place update there is racy in exactly the same way a
        # data-driven array index is -- python accumulates every contribution.
        outer:Optional[Dict[str, object]] = (
            self.outer_vars.get(node.target.id)
            if isinstance(node.target, ast.Name) else None)
        carried:bool = (isinstance(node.target, ast.Name)
                        and self._lookup(node.target.id) is not None
                        and isinstance(self._lookup(node.target.id), tuple)
                        and self._lookup(node.target.id)[0] == "carried")
        racy:bool = ((isinstance(node.target, ast.Subscript)
                      and not self._augassign_is_race_free(node.target))
                     or (outer is not None and bool(outer["writable"]))
                     or carried)
        if racy:
            atomic_name:Optional[str] = self.ATOMIC_ASSIGN_OPS.get(op_name)
            if atomic_name is None:
                where:str = ("this shared outer variable"
                             if outer is not None else "this array")
                self._error(
                    f"'{self._augassign_symbol(op_name)}=' on {where} may be updated by "
                    f"several work items at once, and there is no device atomic operation "
                    f"for it. Write the update explicitly, e.g. use atomic_add for a sum, "
                    f"or accumulate per work item and reduce afterwards "
                    f"(see examples/09_reduction.py)", node)
            if carried:
                # One location shared by every work item, updated through its own buffer.
                # The atomic is the slot's own width (see _bind_carried_scalars, which
                # types the slot from the initializer): an int atomic handed a float
                # value truncated every contribution. The carried marker's second entry
                # is the *name*; the slot's type is the parameter's.
                carried_t:object = "int"
                for parameter_name, parameter_t in self.params:
                    if parameter_name == node.target.id and isinstance(parameter_t, tuple) \
                            and is_ptr(parameter_t):
                        carried_t = parameter_t[1]
                        break
                self._emit_atomic_update(node, op_name, atomic_name, carried_t,
                                         array_code=node.target.id, index_code="0",
                                         pointer_code=node.target.id)
                return
            pointer:Optional[str] = node.target.id if outer is not None else None
            self._visit_atomic_augassign(node, op_name, atomic_name, pointer_code=pointer)
            return

        target_code, target_t = self._visit_target(node.target)
        value_code, value_t = self._visit_expression(node.value)

        # A complex target has no C in-place operator either, but its arithmetic *does* have
        # a device meaning: the header's function, exactly as `_emit_complex_binop` builds
        # it for `z = z + c`. Python reads `z += c` as `z = z + c` when the class defines no
        # `__iadd__` -- the header's record defines none -- so the store is the same call.
        # Without this branch the plain C `+=` reached the compiler and was refused for a
        # struct (`no operator "+=" matches these operands (c128 += c128)`).
        if self._complex_dtype_of(target_t) is not None \
                and op_name in ("Add", "Sub", "Mult", "Div", "Pow"):
            operator_node:ast.BinOp = ast.BinOp(
                left=node.target, op=node.op, right=node.value,
                lineno=node.lineno, col_offset=node.col_offset)
            stored:Translated = self._emit_complex_binop(
                op_name, target_code, target_t, value_code, value_t, operator_node)
            self._emit(f"{target_code} = {stored[0]};")
            return

        # A structure target has no C in-place operator: `v += w` on a class is that
        # class's `__iadd__`, or its `__add__` when it defines no in-place form. The call
        # returns a new value, which is stored back -- exactly what python does.
        if self._is_structure_operand(target_t):
            inplace:Optional[str] = MAGIC_INPLACE.get(op_name)
            owner:Optional[Tuple[type, Any]] = (
                self._magic_owner(target_t, inplace) if inplace is not None else None)
            called:str = inplace or ""
            if owner is None:
                # No in-place method: python answers `v = v + w`, so the plain operator's
                # method (or the refusal that explains there is none) decides.
                magic:Optional[str] = MAGIC_METHODS.get(op_name)
                owner = (self._magic_owner(target_t, magic)
                         if magic is not None else None)
                called = magic or ""
            if owner is not None:
                _cls, method = owner
                call:Translated = self._emit_device_call(
                    called, method, [(target_code, target_t), (value_code, value_t)],
                    node, self_param=_receiver_param_of(method), receiver_as_argument=True)
                self._emit(f"{target_code} = {call[0]};")
                return
            self._error(self._structure_operator_message(op_name, target_t, value_t, node), node)

        # A vector has no C in-place operator either (CUDA rejects `v += w`), so the store
        # is written out as the plain vector operation it means. The right operand goes into
        # a temporary first: the per-component form names it once per component, and a call
        # with side effects (`pixel_color += scene.ray_color(...)`) must be evaluated once.
        vector_store:Optional[str] = self._emit_vector_inplace(
            target_code, op_name, target_code, target_t, value_code, value_t)
        if vector_store is not None:
            if value_code.strip().endswith(")") and "(" in value_code:
                temporary:str = f"_inplace{self._next_temp()}"
                self._emit(f"{value_t} {temporary} = {value_code};")
                vector_store = self._emit_vector_inplace(
                    target_code, op_name, target_code, target_t, temporary, value_t)
            self._emit(vector_store)
            return

        name:str = op_name
        if name == "Div":
            if is_int(target_t) and is_int(value_t):
                self._error("integer /= integer changes type (use // or float operands)", node)
            self._emit(f"{target_code} /= {value_code};")
        elif name == "FloorDiv":
            if is_int(target_t) and is_int(value_t):
                # `a //= b` must floor like python, not truncate like C (see _emit_floored).
                code, _ = self._emit_floored(("floor_div", "floor_mod"), "floor_div",
                                             target_code, target_t, value_code,
                                             value_t, node)
                self._emit(f"{target_code} = {code};")
            else:
                self._emit(f"{target_code} = floor(({target_t})({target_code}) / ({target_t})({value_code}));")
        elif name == "Pow":
            # There is no **= in C; the power is a function call, so the store is
            # written explicitly. An integer target would change type under python's
            # rules, which a C variable cannot do, so only floats are accepted.
            if is_int(target_t) and is_int(value_t):
                self._error("integer **= changes type (use float operands or pow())", node)
            self._emit(f"{target_code} = ({target_t})pow(({target_t})({target_code}), "
                       f"({target_t})({value_code}));")
        elif name in ("LShift", "RShift", "BitOr", "BitAnd", "BitXor", "Add", "Sub", "Mult"):
            symbol:str = {"LShift": "<<", "RShift": ">>", "BitOr": "|", "BitAnd": "&", "BitXor": "^",
                          "Add": "+", "Sub": "-", "Mult": "*"}[name]
            self._emit(f"{target_code} {symbol}= {value_code};")
        elif name == "Mod":
            if is_int(target_t) and is_int(value_t):
                # `a %= b` must carry the divisor's sign like python's `%` (see _emit_floored).
                code, _ = self._emit_floored(("floor_div", "floor_mod"), "floor_mod",
                                             target_code, target_t, value_code,
                                             value_t, node)
                self._emit(f"{target_code} = {code};")
            else:
                # A float remainder is python's floored one too -- and C's `%` is an
                # integer operator anyway, so the augmented spelling goes through the
                # same helper the binary operator uses (see `float_mod` in
                # device_builtins.PYTHON_FLOOR_SOURCES).
                code, _ = self._emit_floored(("floor_div", "floor_mod"), "float_mod",
                                             target_code, promote(target_t, value_t),
                                             value_code, value_t, node)
                self._emit(f"{target_code} = {code};")
        else:
            self._error(f"unsupported operator: {name}", node)


    def _lookup_type_of_index(self, node:ast.AugAssign)->str:
        return "int"
