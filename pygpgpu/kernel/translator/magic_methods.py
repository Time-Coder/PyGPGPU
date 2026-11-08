from __future__ import annotations
import ast
from typing import (Any, Callable, List, Optional, Tuple)
from ..structs import dtype_for as structs_dtype_for
from .notation import (
    DeviceType,
    Translated,
)
from .receivers import (
    MAGIC_COMPARE_REFLECTED,
    MAGIC_REFLECTED,
    _magic_method_of,
    _operator_name,
    _property_accessor,
    _property_carrier,
    _property_of,
    _receiver_param_of,
)
from .vectors import (
    is_vec,
)
from .predicates import (
    is_ptr,
)


class _MagicMethodsMixin:

    def _magic_receiver_type(self, t:DeviceType)->DeviceType:
        """The type a magic method's receiver parameter takes.

        A structure argument travels as a pointer, so a class method's receiver is a
        pointer too; a vector (the package's own types define no magic method) and a scalar
        keep their value type.
        """
        return t


    def _property_descriptor(self, classes:object, name:str,
                             code:str)->Optional[object]:
        """The property descriptor for `name`, from the class that actually owns it.

        The launch knows the exact class of a structure argument (see `_class_of`), and
        that answer wins: two classes with one layout share a dtype, so asking the dtype
        answered with the class that registered it first -- and `b.total` then compiled
        `a`'s getter and returned its value.
        """
        exact:Optional[type] = self._class_of(code, None)
        if exact is not None:
            descriptor:Optional[object] = _property_of(exact, name)
            if descriptor is not None:
                return descriptor
        return _property_carrier(classes, name)


    def _emit_property_write(self, cls:Optional[type], name:str, base:Translated,
                             value:ast.expr, node:ast.AST)->bool:
        """`box.total = v` as a device call to the property's setter, or False.

        A setter is an ordinary method taking the new value, so the assignment becomes a
        call statement of its own. False when the class defines no such property, which
        leaves the field store to answer.
        """
        descriptor:Optional[object] = self._property_descriptor(cls, name, base[0])
        if descriptor is None:
            return False
        setter:Optional[Callable] = _property_accessor(descriptor, writer=True)
        if setter is None:
            self._error(
                f"property '{name}' has no setter: give the class a '@{name}.setter'",
                node)
        value_code, value_t = self._visit_expression(value)
        call:Translated = self._emit_device_call(
            name, setter, [base, (value_code, value_t)], node,
            self_param=_receiver_param_of(setter), receiver_as_argument=True)
        self._emit(f"{call[0]};")
        # The store happens inside the setter, which the static analysis cannot see: it
        # reads `h.doubled = v` but the generated statement is a call, so the parameter
        # would stay read-only and the host would drop the write. `atomic_targets` is the
        # force-writable channel a member store already uses.
        if isinstance(value, ast.expr) and isinstance(base[0], str):
            written_name:str = base[0].replace("&", "").replace("(", "").replace(")", "")
            if written_name.isidentifier():
                self.atomic_targets.add(written_name)
                self.member_writes.add(written_name)
        return True


    def _emit_property_read(self, cls:Optional[type], name:str, base:Translated,
                            node:ast.AST)->Optional[Translated]:
        """`box.total` as a device call to the property's getter, or None.

        A property is a member the class computes, so the read is a method call with the
        receiver as its argument -- the same shape a plain method call takes. None when the
        class defines no such property, which leaves the field lookup to answer.
        """
        descriptor:Optional[object] = self._property_descriptor(cls, name, base[0])
        if descriptor is None:
            return None
        getter:Optional[Callable] = _property_accessor(descriptor, writer=False)
        if getter is None:
            self._error(f"property '{name}' has no getter", node)
        return self._emit_device_call(
            name, getter, [base], node,
            self_param=_receiver_param_of(getter), receiver_as_argument=True)


    def _emit_magic_conversion(self, magic:str, code:str, t:object,
                               node:ast.AST)->Optional[Translated]:
        """`int(box)` / `abs(box)` and friends as a device call to the class's method.

        A class that defines the conversion gives the builtin a meaning the device keeps:
        casting an aggregate is not what the caller wrote. None when the operand carries
        no such method, which leaves the ordinary scalar path to answer.
        """
        owner:Optional[Tuple[type, Any]] = self._magic_owner(t, magic)
        if owner is None:
            return None
        _cls, method = owner
        return self._emit_device_call(
            magic, method, [(code, t)], node,
            self_param=_receiver_param_of(method), receiver_as_argument=True)


    def _emit_magic_binop(self, magic:str, left:str, lt:object, right:str, rt:object,
                          node:ast.BinOp)->Optional[Translated]:
        """`a op b` as a device call to the class's magic method, or None.

        Python resolves `a + b` as `a.__add__(b)`, and then -- if the left class does not
        define it -- as `b.__radd__(a)`. Both spellings are kept: the reflected form is how a
        scalar on the left reaches a class that only knows how to multiply itself by one
        (`2.0 * v` calls `v.__rmul__(2.0)`), which is a shape a kernel body does write.
        """
        owner:Optional[Tuple[type, Any]] = self._magic_owner(lt, magic)
        if owner is not None:
            _cls, method = owner
            return self._emit_device_call(
                magic, method, [(left, lt), (right, rt)], node,
                self_param=_receiver_param_of(method), receiver_as_argument=True)

        reflected:Optional[str] = MAGIC_REFLECTED.get(_operator_name(magic))
        if reflected is not None:
            owner = self._magic_owner(rt, reflected)
            if owner is not None:
                _cls, method = owner
                # The reflected method takes the right operand's receiver first, so the
                # operands are handed over in the order the call has them.
                return self._emit_device_call(
                    reflected, method, [(right, rt), (left, lt)], node,
                    self_param=_receiver_param_of(method), receiver_as_argument=True)
        return None


    def _emit_magic_compare(self, magic:str, left:str, lt:object, right:str, rt:object,
                            op:ast.cmpop)->Optional[Translated]:
        """`a == b` as a device call to the class's comparison, or None."""
        # A containment test asks the RIGHT operand: `x in box` is `box.__contains__(x)`.
        # Its result is a truth value, so `not in` negates the same call. A chained form
        # (`x in a in b`) is refused above, where the operator is read.
        if magic == "__contains__":
            owner:Optional[Tuple[type, Any]] = self._magic_owner(rt, magic)
            if owner is not None:
                _cls, method = owner
                call:Translated = self._emit_device_call(
                    magic, method, [(right, rt), (left, lt)], op,
                    self_param=_receiver_param_of(method), receiver_as_argument=True)
                if op.__class__.__name__ == "NotIn":
                    return f"(!{call[0]})", "bool"
                return call
            return None

        owner:Optional[Tuple[type, Any]] = self._magic_owner(lt, magic)
        if owner is not None:
            _cls, method = owner
            return self._emit_device_call(
                magic, method, [(left, lt), (right, rt)], op,
                self_param=_receiver_param_of(method), receiver_as_argument=True)

        # Python reads `a < b` as `b.__gt__(a)` when the left class has no `__lt__`.
        reflected:Optional[str] = MAGIC_COMPARE_REFLECTED.get(_operator_name(magic))
        if reflected is not None:
            owner = self._magic_owner(rt, reflected)
            if owner is not None:
                _cls, method = owner
                return self._emit_device_call(
                    reflected, method, [(right, rt), (left, lt)], op,
                    self_param=_receiver_param_of(method), receiver_as_argument=True)
        return None


    def _class_of(self, code:str, t:object)->Optional[type]:
        """The class a value was built from, when this launch knows it exactly.

        One dtype can stand for several classes -- two classes with one layout share it --
        so a dtype alone cannot say whose method `p.total` is. The call does know: the
        caller handed over an instance, and its type travels in `struct_arg_classes` under
        the parameter's name. That answer wins over the dtype, which only has the class
        that registered the layout first.
        """
        name:str = code.strip().lstrip("&").strip("()").strip()
        exact:Optional[type] = self.struct_arg_classes.get(name)
        return exact


    def _magic_owner(self, t:object, magic:str,
                     code:str = "")->Optional[Tuple[type, Any]]:
        """The class whose magic method an operand's operator should call, and that method.

        A python class that defines `__add__` gives `a + b` a meaning of its own, and a
        device translation has to keep it: the operator becomes a device call to that
        method, with the operands as its arguments, exactly as `ray.at(t)` becomes a call to
        `Ray_at`. Returns None when the operand carries no such method, which leaves the
        ordinary operator path (a scalar or a vector operation) to answer.
        """
        from .. import structs as _structs_module  # noqa: PLC0415

        base:str = t[1] if isinstance(t, tuple) and is_ptr(t) else t
        if not isinstance(base, str):
            return None
        dtype:object = _structs_module.dtype_for(base)
        if dtype is None:
            return None
        # The launch may know the exact class of this value (see `_class_of`); that answer
        # is the one that counts, because two classes sharing a layout would otherwise be
        # answered by whichever registered first.
        exact:Optional[type] = self._class_of(code, t) if code else None
        if exact is not None:
            method:object = _magic_method_of(exact, magic)
            if method is not None:
                return exact, method
        # Otherwise ask every class the layout carries -- a shared layout means the first
        # registered one is not necessarily the class whose method this is.
        for cls in _structs_module.declared_classes(dtype):
            method:object = _magic_method_of(cls, magic)
            if method is not None:
                return cls, method
        return None


    def _is_structure_operand(self, t:object)->bool:
        """Whether a translated type is a structure value (a quaternion is one too).

        A structure argument travels as a pointer, so the pointer type names one as well.
        """
        if isinstance(t, tuple) and is_ptr(t):
            return structs_dtype_for(t[1]) is not None
        if not isinstance(t, str) or is_vec(t):
            return False
        return structs_dtype_for(t) is not None


    def _structure_operator_message(self, operator:str, lt:object, rt:object,
                                    node:ast.AST)->str:
        """Why a structure cannot take part in an operator, and what to write instead."""
        from .. import structs  # noqa: PLC0415

        operand:object = lt if self._is_structure_operand(lt) else rt
        dtype:object = structs.dtype_for(operand)
        members:str = ", ".join(dtype.names or ()) if dtype is not None else "?"
        symbol:str = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "Mod": "%",
                      "Pow": "**", "FloorDiv": "//", "Lt": "<", "LtE": "<=", "Gt": ">",
                      "GtE": ">=", "Eq": "==", "NotEq": "!="}.get(operator, operator)
        return (
            f"a structure cannot be used with the '{symbol}' operator: device code has no "
            f"arithmetic for a structure value, and the operands here are {lt} and {rt}. "
            f"Operate on the members instead (`s.{members.split(', ')[0]} {symbol} ...`), or "
            f"move the computation into a host method before the kernel -- which is how a "
            f"quaternion rotation belongs, unless a dedicated device function for it exists")


    def _emit_self_method_call(self, kind:str, target:object, attr:str,
                               args:List[Translated], node:ast.Call)->Translated:
        """Call self.<attr>(...) where the member is a method of the kernel's own class."""
        if getattr(target, "is_pykernel", False):
            self._error(
                f"self.{attr} is a @kernel method and cannot be called from device code; "
                f"move the shared logic into a plain method or function", node)
        return self._emit_device_call(attr, target, args, node,
                                      self_param=self.self_param if kind == "method" else "")


    def _why_not_callable(self, kind:str, name:str)->str:
        """Explain why a self.xxx member cannot be called, for the error message."""
        if kind == "classmethod":
            return "a classmethod (cls has no meaning on a device)"
        if kind == "kernel":
            return "a @kernel method (kernels are launched from the host, not called from device code)"
        return ("not a method or static method of the class; only those, and module-level "
                "functions, can be called from device code")
