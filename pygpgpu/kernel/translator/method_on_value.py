# A method call on a *value* the body holds: ay.at(t)\, ecv.inst(v)\ --
# the receiver is the value's own expression, the callee's first parameter binds
# it, and the class the method is translated from is the one the value's dtype
# declares (or the vector registry's, for the vector-typed values).
from __future__ import annotations

import ast
from typing import Optional

import numpy as np

from .notation import PTR, Translated
from .predicates import is_ptr
from .receivers import self_attr_kind
from .vectors import _vec_registry


class _MethodOnValueMixin:
    def _emit_method_on_value(self, func:ast.Attribute, node:ast.Call)->Translated:
        """`value.method(args)` as a device call whose first argument is the value.

        A structure's methods are ordinary python functions defined in the class the caller
        declared, and their first parameter is the receiver -- which is exactly what `self`
        already is in a device helper. So there is nothing new to translate: the call becomes
        a device function whose first argument is the receiver expression, and the body then
        reads `self.origin` the way any structure-typed parameter would.

        The class has to be reachable, which is why only a `ctypes.Structure` subclass
        qualifies: a bare record dtype has no methods, and two classes with one layout share
        a dtype, so the dtype alone cannot say whose methods to compile.
        """
        from .. import structs

        base_code, base_t = self._visit_expression(func.value)
        # A complex value's methods are numpy's, and the device header spells the same
        # operations: `.conjugate()` flips the imaginary part, and the rest are the free
        # functions a reader may reach for through the value itself.
        complex_dtype:Optional[np.dtype] = self._complex_dtype_of(base_t)
        if complex_dtype is not None:
            translated:Optional[Translated] = self._emit_complex_method(
                func.attr, base_code, base_t, complex_dtype, node)
            if translated is not None:
                return translated
            self._error(
                f"a complex value has no method '.{func.attr}()': numpy's complex scalars"
                f" define 'conjugate()' and 'conj()', and the rest are the module functions"
                f" (np.sqrt, np.abs, np.angle, ...)", node)
        receiver_is_pointer:bool = (isinstance(base_t, tuple) and is_ptr(base_t))
        base_struct:str = base_t[1] if receiver_is_pointer else base_t
        owner:Optional[type] = None
        if isinstance(base_struct, str):
            dtype:Optional[np.dtype] = structs.dtype_for(base_struct)
            if dtype is not None:
                # The class that actually defines the method: two callers can declare
                # classes with one layout, and only one of them carries this attribute.
                owner = (structs.declared_class_carrying(dtype, func.attr)
                         or structs.declared_class(dtype))
            elif base_struct in _vec_registry():
                owner = _vec_registry()[base_struct]
        if owner is None:
            self._error(
                f"cannot call '.{func.attr}()' on a value of type {base_t}: methods are "
                f"translated from the class they are defined on, and only a "
                f"ctypes.Structure subclass or a vector type carries one", node)

        kind:str
        target:object
        kind, target = self_attr_kind(owner, func.attr)
        if kind == "method":
            args = [self._visit_expression(a) for a in node.args]
            # A method that assigns to its receiver's members needs the receiver *by
            # address*: a by-value structure is a copy, so `self.normal = ...` inside the
            # method would be lost. `HitRecord.set_face_normal` is exactly that shape, and
            # with a by-value receiver every normal stayed zero -- the GPU drew a uniform
            # grey ball while the interpreter, which runs the method unchanged, was right.
            receiver_param:str = (target.__code__.co_varnames[0]
                                  if getattr(target, "__code__", None) else "self")
            if receiver_is_pointer:
                # A structure argument already travels as a pointer: hand it over as it is,
                # whether the callee writes its members or only reads them. The receiver's
                # parameter name is named so the callee binds it as the structure it is --
                # without it, `self.origin` inside the method would read the value form.
                return self._emit_device_call(
                    func.attr, target, [(base_code, base_t)] + args, node,
                    self_param=receiver_param, receiver_as_argument=True)
            # A value receiver is handed over by address: every structure parameter is a
            # pointer (so `self.<member>` reads the one form), and a local value's address
            # is available at the call site. The copy python makes for a by-value call is
            # the one thing that cannot be kept, and a method that assigns to its receiver
            # needs the address for the write to reach the caller.
            return self._emit_device_call(
                func.attr, target,
                [(f"&({base_code})", (PTR, base_struct))] + args, node,
                self_param=receiver_param, receiver_as_argument=True)
        if kind == "staticmethod":
            args = [self._visit_expression(a) for a in node.args]
            return self._emit_device_call(func.attr, target, args, node)
        self._error(
            f"cannot call '.{func.attr}()' on {owner.__name__}: it is "
            f"{self._why_not_callable(kind, func.attr)}", node)


