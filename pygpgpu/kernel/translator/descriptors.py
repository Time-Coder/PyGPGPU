"""A class attribute that computes its value: a descriptor's `__get__`/`__set__` as device calls.

A class body may hold a *descriptor* rather than a field. Reading `box.total` runs
`type(box).total.__get__(box, type(box))`, and storing `box.total = v` runs `__set__`; python
resolves either before it looks for a field, so a class that computes a member keeps that
meaning on the device. `property` is the descriptor the translator already answered with a
direct getter/setter call (`_emit_property_read`/`_emit_property_write`), and it stays there:
its accessors take only the instance, which is exactly what a device call can carry.

Every other descriptor has two parameters a device cannot hold: the descriptor *object* (a
class attribute) and the `owner` *class*. So `__get__`/`__set__` are translated with those
parameters skipped (`leading_skips` on the callee translation), and a body that reads one is
refused by name rather than bound to a value that does not exist. A descriptor that keeps
state in itself (`self.value`) is therefore host-only, and the refusal says so.

A `ctypes` field is a descriptor too (`CField`), and its `__get__` *is* the member read; it is
excluded here so a plain field stays a field.
"""
from __future__ import annotations
import ast
import inspect
from typing import (List, Optional)
from ..structs import (
    declared_classes as structs_declared_classes,
    dtype_for as structs_dtype_for,
)
from .notation import (
    DeviceType,
    Translated,
)
from .predicates import (
    is_ptr,
)
from .source_text import (
    get_dedented_source,
    _parse_kernel_source,
)


def descriptor_of(cls:type, name:str)->Optional[object]:
    """The class-body attribute `cls` defines for `name` when it is a descriptor, or None.

    `property` answers None: its accessors are the ones the property path already calls, and
    it names the instance alone. A method (`inspect.isfunction`), a `staticmethod`/
    `classmethod`, a `@kernel` and a `ctypes` field are all excluded for their own reasons --
    the first three are members the call paths answer, and a `CField` *is* the field read.

    Module-level because the launch asks the same question before it packs the receiver: a
    descriptor on the kernel's own class is computed by the device, so its name must not also
    be packed as a member (see `decorator/_phase1`).
    """
    own:object = getattr(cls, "__dict__", {}).get(name)
    if own is None or isinstance(own, property):
        return None
    if isinstance(own, (staticmethod, classmethod)) or inspect.isfunction(own):
        return None
    if getattr(own, "is_pykernel", False):
        return None
    if type(own).__module__ == "ctypes":
        return None
    accessor_type:type = type(own)
    if not (hasattr(accessor_type, "__get__") or hasattr(accessor_type, "__set__")):
        return None
    return own


class _DescriptorMixin:

    def _descriptor_of(self, cls:type, name:str)->Optional[object]:
        """The descriptor `cls` defines for `name`, or None (see `descriptors.descriptor_of`)."""
        return descriptor_of(cls, name)


    def _descriptor_carrier(self, classes:Optional[List[type]],
                            name:str)->Optional[object]:
        """The descriptor for `name` from the first of `classes` that defines one.

        One dtype can stand for several classes -- two classes with one layout share it -- so
        the descriptor has to be looked for in every class the layout carries, not just the
        one registered first.
        """
        for cls in classes or []:
            descriptor:Optional[object] = self._descriptor_of(cls, name)
            if descriptor is not None:
                return descriptor
        return None


    def _descriptor_accessor(self, descriptor:object, writer:bool)->Optional[object]:
        """The `__get__` (or `__set__`) function of a descriptor, or None when it has none."""
        function:object = getattr(type(descriptor), "__set__" if writer else "__get__", None)
        return function if inspect.isfunction(function) else None


    def _descriptor_parameter_error(self, accessor:object, name:str, writer:bool,
                                    node:ast.AST)->Optional[str]:
        """Why a descriptor's accessor cannot be translated, or None when it can.

        Two parameters of a descriptor protocol have no device value: the descriptor object
        itself, and -- for `__get__` -- its owner class. They are skipped when the accessor is
        translated, so a body that reads one is refused by name instead of leaving a bare
        "unknown name". (`__set__`'s instance and value *are* the call's arguments.)
        """
        code:object = getattr(accessor, "__code__", None)
        if code is None:
            return None
        parameters:List[str] = list(code.co_varnames[:code.co_argcount])
        if not parameters:
            return None
        forbidden:List[str] = [parameters[0]]
        if not writer and len(parameters) >= 3:
            forbidden.append(parameters[2])
        try:
            tree:ast.Module = _parse_kernel_source(get_dedented_source(accessor))
        except (OSError, TypeError, SyntaxError):
            return None
        used:set = {child.id for child in ast.walk(tree)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}
        for parameter in forbidden:
            if parameter in used:
                what:str = ("the descriptor object itself" if parameter == parameters[0]
                            else "its owner class")
                return (
                    f"the descriptor '{name}' reads '{parameter}' in its accessor, which "
                    f"names {what}: that is a host object, so device code has no value for "
                    f"it. Compute the value from the instance instead (a descriptor that "
                    f"keeps state in itself stays host-only)")
        return None


    def _classes_of_struct_type(self, t:DeviceType)->List[type]:
        """The classes a translated type's structure layout carries."""
        struct_t:object = t[1] if isinstance(t, tuple) and is_ptr(t) else t
        if not isinstance(struct_t, str):
            return []
        dtype:object = structs_dtype_for(struct_t)
        if dtype is None:
            return []
        return structs_declared_classes(dtype)


    def _descriptor_classes_of(self, expr:ast.expr, base_t:DeviceType)->List[type]:
        """The classes to ask for a descriptor, for an attribute base expression.

        A receiver (`self.attr`) is the kernel's own instance: its class is known directly
        (`self_owner`), while the receiver's synthesized layout may carry no registered class
        at all -- `register_name_hint` names the C type but only a *value* receiver is also
        registered as a declaring class.
        """
        classes:List[type] = self._classes_of_struct_type(base_t)
        if (self.self_owner is not None and self.self_param
                and isinstance(expr, ast.Name) and expr.id == self.self_param):
            return [self.self_owner] + [cls for cls in classes if cls is not self.self_owner]
        return classes


    def _emit_descriptor_read(self, classes:Optional[List[type]], name:str,
                              base:Translated, node:ast.AST)->Optional[Translated]:
        """`box.attr` as the class's own `__get__`, or None for a plain member."""
        descriptor:Optional[object] = self._descriptor_carrier(classes, name)
        if descriptor is None:
            return None
        accessor:Optional[object] = self._descriptor_accessor(descriptor, writer=False)
        if accessor is None:
            return None
        refusal:Optional[str] = self._descriptor_parameter_error(accessor, name, writer=False,
                                                                 node=node)
        if refusal is not None:
            self._error(refusal, node)
        # `__get__(self, instance, owner=None)`: the descriptor and the owner are skipped,
        # and the instance is the one argument a device can carry.
        return self._emit_device_call(
            f"__get__ of '{name}'", accessor, [base], node, leading_skips=1)


    def _emit_descriptor_write(self, classes:Optional[List[type]], name:str, base:Translated,
                               base_node:ast.expr, value_code:str, value_t:object,
                               node:ast.AST)->bool:
        """`box.attr = v` as the class's own `__set__`, or False when it is a plain member."""
        descriptor:Optional[object] = self._descriptor_carrier(classes, name)
        if descriptor is None:
            return False
        accessor:Optional[object] = self._descriptor_accessor(descriptor, writer=True)
        if accessor is None:
            return False
        refusal:Optional[str] = self._descriptor_parameter_error(accessor, name, writer=True,
                                                                 node=node)
        if refusal is not None:
            self._error(refusal, node)
        call:Translated = self._emit_device_call(
            f"__set__ of '{name}'", accessor, [base, (value_code, value_t)], node,
            leading_skips=1)
        self._emit(f"{call[0]};")
        # The store happens inside the method, so nothing in the generated statement is a
        # member write; marking the instance keeps its parameter writable (and read back),
        # the same force-writable channel a property setter uses.
        self._record_member_write(base_node)
        return True
