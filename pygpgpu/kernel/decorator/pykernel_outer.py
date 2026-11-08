from __future__ import annotations
from typing import (
    Dict,
    List,
    Optional,
)
import numpy as np
from ..translator import (
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
        for name, outer in outer_vars.items():
            kind:str = outer["kind"]
            value:object = self._read_outer(name, kind, outer.get("scope"))
            if is_ptr(outer["type"]):
                if value is None:
                    raise ValueError(
                        f"outer array '{name}' is undefined at launch time")
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
        return {"args": args, "writable": writable, "full": full}


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
