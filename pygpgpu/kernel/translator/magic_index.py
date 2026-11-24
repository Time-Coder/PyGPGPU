"""A class value's subscripts: `box[i]` and `box[i] = v` are the class's own methods.

Split out of `magic_methods.py` by subject: the operator/truth dispatch answers what a value
*means*, while this module answers how it is *indexed*. Both are magic-method dispatch, but
the index pair is reached from two other mixins (`subscripts.py` for the read,
`assign.py`/`augassign.py` for the store) and reads as one idea, so it is one file.
"""
from __future__ import annotations
import ast
from typing import (Optional)
from ..structs import dtype_for as structs_dtype_for
from .notation import (
    DeviceType,
    Translated,
)
from .predicates import (
    is_ptr,
)
from .magic_tables import (
    MagicOwner,
)
from .receivers import (
    _receiver_param_of,
)


class _MagicIndexMixin:

    def _class_index_base(self, node:ast.expr)->Optional[Translated]:
        """The class value an expression stands for, when it is subscripted as one.

        `box[i]` is `box.__getitem__(i)` when `box` is a class instance, and the same holds
        for a member that holds one (`self.tri[i]`, `a.inner[i]`): the class's own method
        answers, not an array index. An array parameter, a work-group buffer and an image
        answer None, so their own index paths keep running. Only a name and a member read
        are resolved here -- a call or a nested subscript has no device value this could
        safely evaluate a second time.
        """
        if isinstance(node, ast.Name):
            binding:DeviceType = self._lookup(node.id)
            if binding is None:
                return None
            if isinstance(binding, tuple) and is_ptr(binding):
                # A one-element record array is how a structure argument travels, and an
                # array of records is the same pointer type; the class's method decides
                # which of the two the caller meant, exactly as the read path always has.
                return (self._visit_expression(node)
                        if isinstance(binding[1], str)
                        and structs_dtype_for(binding[1]) is not None else None)
            if self._is_structure_operand(binding):
                return self._visit_expression(node)
            return None
        if isinstance(node, ast.Attribute):
            translated:Translated = self._visit_expression(node)
            return translated if self._is_structure_operand(translated[1]) else None
        return None


    def _emit_magic_getitem(self, node:ast.Subscript)->Optional[Translated]:
        """`box[i]` as the class's own `__getitem__`, or None for a non-class base.

        None is the answer for an array element and for a class that defines no
        `__getitem__`: both leave the ordinary index path (or its refusal) to answer, which
        is what `record[i]` on a structured array needs.
        """
        base:Optional[Translated] = self._class_index_base(node.value)
        if base is None:
            return None
        owner:Optional[MagicOwner] = self._magic_owner(
            base[1], "__getitem__", base[0])
        if owner is None:
            return None
        _cls, method = owner
        index_code, index_t = self._visit_expression(node.slice)
        return self._emit_device_call(
            "__getitem__", method, [base, (index_code, index_t)], node,
            self_param=_receiver_param_of(method), receiver_as_argument=True)


    def _emit_magic_setitem(self, target:ast.Subscript, value_code:str, value_t:object,
                            node:ast.AST)->bool:
        """`box[i] = v` as the class's own `__setitem__`, or False for a non-class base.

        The store happens inside the method, so nothing in the generated statement is a
        member write. The receiver is marked writable anyway -- the same force-writable
        channel a property setter uses -- because the static read/write analysis sees a
        call: without it the parameter stays `const` and the callee's own store is a
        compile error.
        """
        base:Optional[Translated] = self._class_index_base(target.value)
        if base is None:
            return False
        owner:Optional[MagicOwner] = self._magic_owner(
            base[1], "__setitem__", base[0])
        if owner is None:
            return False
        _cls, method = owner
        index_code, index_t = self._visit_expression(target.slice)
        call:Translated = self._emit_device_call(
            "__setitem__", method,
            [base, (index_code, index_t), (value_code, value_t)], node,
            self_param=_receiver_param_of(method), receiver_as_argument=True)
        self._emit(f"{call[0]};")
        self._record_member_write(target.value)
        return True
