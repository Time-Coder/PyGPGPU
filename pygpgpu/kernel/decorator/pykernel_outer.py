from __future__ import annotations
import ctypes
from typing import (
    Dict,
    List,
    Optional,
)
import numpy as np
from ..translator import (
    OuterObjectEntry,
    is_ptr,
)
from .slots import (
    _kernel_scalar_argument,
    _outer_scope_value,
    _outer_slot_dtype,
    _scalar_slot_value,
)
from ._typing import (
    OuterState,
)


class _PykernelOuterMixin:
    def _bind_outer_values(self, outer_vars:Dict[str, object],
                           backend:str)->OuterState:
        """The launch arguments carrying this call's outer-scope values.

        A read-only outer variable is passed by value, refreshed on every call, so a
        changed scope value is seen immediately -- that is the whole point of
        parameterizing it instead of inlining it. A declared `nonlocal`/`global` scalar
        travels as a one-element buffer, because the device writes through the pointer;
        the buffer is copied back into the scope once the launch has completed (see
        _write_back_outer and _with_returns). An outer array is passed as itself, and the
        backend's read/write analysis reads it back when the kernel wrote it.
        """
        args:List[object] = []
        full:List[object] = []
        writable:List[Dict[str, object]] = []
        objects:List[Dict[str, object]] = []
        for name, outer in outer_vars.items():
            kind:str = outer["kind"]
            value:object = self._read_outer(name, kind, outer.get("scope"))
            if is_ptr(outer["type"]):
                if value is None:
                    raise ValueError(
                        f"outer array '{name}' is undefined at launch time")
                spec:Optional[OuterObjectEntry] = outer.get("object")
                if spec is not None:
                    # A plain python object travels as the structure the translator
                    # synthesized; the members the body writes are copied home after the
                    # launch through the same binding object arguments use.
                    instance:object = _outer_object_instance(value, spec)
                    args.append(instance)
                    full.append(instance)
                    objects.append({"name": name, "source": value,
                                    "instance": instance,
                                    "written": list(spec["written"])})
                    continue
                args.append(value)
                full.append(value)
                continue
            if not outer["writable"]:
                converted:object = _kernel_scalar_argument(value, outer["type"])
                args.append(converted)
                full.append(converted)
                continue
            buffer:np.ndarray = np.zeros(1, dtype=_outer_slot_dtype(outer["type"]))
            buffer[0] = _scalar_slot_value(value, outer["type"])
            args.append(buffer)
            full.append(buffer)
            writable.append({"name": name, "kind": kind, "buffer": buffer,
                             "type": outer["type"]})
        return {"args": args, "writable": writable, "full": full, "objects": objects}


    def _read_outer(self, name:str, kind:str,
                    scope:Optional[Dict[str, object]] = None)->object:
        """The current value of an outer name, read the way python would resolve it.

        ``scope`` is the namespace the translator resolved the name in, which is the
        kernel's module for a name the body reads itself and the *helper's* module for one
        a device helper reads. Both are read live so a changed value is seen on the next
        call, which is why the namespace is carried rather than the value.
        """
        if kind == "closure":
            cells:Dict[str, object] = {}
            for cell_name, cell in zip(self.raw_func.__code__.co_freevars,
                                       self.raw_func.__closure__ or ()):
                cells[cell_name] = cell
            cell = cells.get(name)
            if cell is None:
                raise ValueError(f"outer variable '{name}' is not in the kernel's closure")
            return cell.cell_contents
        if scope is not None:
            return scope.get(name, None)
        return self.raw_func.__globals__.get(name, None)


    def _write_back_outer(self, outer_state:OuterState)->None:
        """Copy a declared outer scalar's final value back into its scope."""
        for entry in outer_state["writable"]:
            value:object = _outer_scope_value(entry["buffer"][0], entry["type"])
            self._store_outer(entry["name"], entry["kind"], value)


    def _write_back_receiver(self, plan:object)->None:
        """Copy the receiver struct's members home, onto the caller's receiver.

        A body that assigns `self.<attr> = ...` (or `cls.<attr> = ...`) stores through the
        receiver struct's device buffer; the backend's read-back lands in the holder the
        launch packed (`plan.holder`, a ctypes structure). Python's own semantics for the
        same statement is an attribute write on the receiver -- the instance for an
        instance method, the *class* for a classmethod (`cls.attr = v` rebinds the class
        attribute). This copies each member the launch carried back onto that object,
        which is what makes the store observable to the caller.

        A member that is itself a `ctypes` structure (a nested python object: python
        semantics says `self.inner.x = v` mutates the object `self.inner` already names)
        is copied **in place**, field by field, so the caller's original object stays the
        one on the attribute -- a second python name for it (an alias) sees the update,
        which is exactly what `self.inner.x = v` means and what replacing the attribute
        would silently break.

        The launch recorded the pairing on the plan; an asynchronous launch's settle
        callback runs after the phases have let the plan go, so it passes None and the
        pairing the launch left on the kernel is used instead.
        """
        pairing:object = (getattr(plan, "receiver_write_back", None)
                          if plan is not None else None)
        if pairing is None:
            pairing = getattr(self, "_receiver_write_back", None)
        if not isinstance(pairing, tuple) or len(pairing) != 3:
            return
        source:object = pairing[0]
        holder:object = pairing[1]
        members:List[str] = pairing[2]
        for member in members:
            value:object = getattr(holder, member, None)
            if value is None:
                continue
            current:object = getattr(source, member, None)
            if (isinstance(value, ctypes.Structure) and isinstance(current, ctypes.Structure)
                    and type(value)._fields_ == type(current)._fields_):
                # The same layout, and the caller's object must stay the one the
                # attribute names: copy the fields through in place.
                _copy_structure_fields(value, current)
            else:
                setattr(source, member, value)

    def _store_outer(self, name:str, kind:str, value:object)->None:
        """Assign an outer name, through its closure cell or the module globals."""
        if kind == "closure":
            for cell_name, cell in zip(self.raw_func.__code__.co_freevars,
                                       self.raw_func.__closure__ or ()):
                if cell_name == name:
                    cell.cell_contents = value
                    return
            raise ValueError(f"outer variable '{name}' is not in the kernel's closure")
        self.raw_func.__globals__[name] = value


def _copy_structure_fields(source:object, target:object)->None:
    """One ctypes structure's fields onto another of the same layout, recursively.

    A nested structure field recurses the same way (in place), so a store two levels
    deep (`self.inner.point.x = v`) still reaches the object the caller's attribute
    already names; every other field is a plain attribute write, which for a ctypes
    substructure field would rebind it -- a nested field is therefore reached through
    the recursion, never through setattr.
    """
    for field, kind in type(target)._fields_:
        value:object = getattr(source, field)
        if isinstance(kind, type) and issubclass(kind, ctypes.Structure):
            _copy_structure_fields(value, getattr(target, field))
        elif isinstance(kind, type) and issubclass(kind, ctypes.Array):
            _copy_array_fields(value, getattr(target, field))
        else:
            setattr(target, field, value)


def _copy_array_fields(source:object, target:object)->None:
    """A fixed-length ctypes array's elements, recursing into structure elements."""
    for index in range(len(target)):
        value:object = source[index]
        element:object = target[index]
        if isinstance(element, ctypes.Structure):
            _copy_structure_fields(value, element)
        elif isinstance(element, ctypes.Array):
            _copy_array_fields(value, element)
        else:
            target[index] = value


def _outer_object_instance(source:object, spec:OuterObjectEntry)->object:
    """The structure instance a launch hands the device for one outer python object."""
    structure:type = spec["structure"]
    values:Dict[str, object] = {}
    for member in spec["members"]:
        values[member] = getattr(source, member)
    return structure(**values)
