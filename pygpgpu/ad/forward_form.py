# The forward-mode form: `k.jvp(tangent=dx)(args)`.
#
# Split from the reverse forms by subject: forward mode propagates tangents in one pass
# and needs no accumulation shape, no scatter and no seed -- its builder is
# `forward.JVPBuilder` and its output rides on the direction arrays the caller binds.
# The configuring/launch split it inherits lives with the other forms.
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .build import (
    RETURN_SPELL,
    _build_jvp_kernel,
    _check_tracked_members,
    _scan_of,
    _zeros_like,
)
from .body_scan import BodyScan
from .forms import _ADForm
from . import rules as rule_table


class JVPForm(_ADForm):
    """`k.jvp(...)` -- the fused forward form: tangents travel beside the values."""

    def __init__(self, kernel:object, instance:object = None,
                 backend:Optional[str] = None, wrt:Optional[Sequence[str]] = None,
                 tangent:object = None, out:object = RETURN_SPELL):
        self.tangent:object = tangent
        self.out:object = out
        super().__init__(kernel, instance, backend, wrt)

    def _has_config(self)->bool:
        return self.tangent is not None or self.out != RETURN_SPELL

    def _spell(self)->str:
        return "k.jvp"

    def _config_keys(self)->Set[str]:
        return {"tangent", "out", "wrt"}

    def _copy(self)->"_ADForm":
        form:JVPForm = JVPForm(self.kernel, self.instance, self.backend, self.wrt,
                               self.tangent, self.out)
        form.fresh = self.fresh
        return form

    def _launch_form(self, args:Tuple[object, ...], kwargs:Dict[str, object],
                     launch:str = "call")->object:
        if self.tangent is None:
            raise self._error("`tangent=` binds the input direction `dx`: pass an array "
                              "of the tracked argument's shape")
        arguments:List[Tuple[str, object]] = self._bind(args, kwargs)
        _check_tracked_members(self, arguments)
        wrt:List[str] = self._resolve_wrt(arguments)
        values:Dict[str, object] = dict(arguments)
        directions:Dict[str, object] = self._resolve_tangents(wrt)
        out_name:Optional[str] = self._resolve_out_array(values)
        tangents:Dict[str, str] = {tracked: f"_t_{tracked}" for tracked in wrt}
        out_tangent:Optional[str] = f"_t_{out_name}" if out_name is not None else None
        extra:List[object] = [directions[tracked] for tracked in wrt]
        adjoint:object = self._adjoint(self._key(wrt, out_name), lambda: self._build(
            wrt, tangents, out_name, out_tangent))
        if out_tangent is not None:
            buffer:object = _zeros_like(values[out_name])
            extra.append(buffer)
            self._launch(adjoint, self._positional(arguments), extra, launch=launch)
            # The form answers the buffer it allocated: the launch wrote the tangent into
            # it, and the caller reads it (after the future lands, for a deferred call).
            return buffer
        if launch != "call":
            raise self._error(
                "this kernel answers a scalar through its return slot, which a deferred "
                "launch cannot carry: name `out=` with an array result, or use the "
                "synchronous spelling")
        return self._launch(adjoint, self._positional(arguments), extra)

    def _key(self, wrt:List[str], out_name:Optional[str])->Tuple[object, ...]:
        return ("jvp", tuple(wrt), self.out, out_name, rule_table.RULES_VERSION,
                tuple(sorted(self.kernel._type_map.items())))

    def _resolve_tangents(self, wrt:List[str])->Dict[str, object]:
        """The direction per tracked parameter, from the `tangent=` binding."""
        provided:object = self.tangent
        if isinstance(provided, dict):
            missing:Set[str] = set(wrt) - set(provided)
            if missing:
                raise self._error(f"`tangent=` names no direction for {sorted(missing)}")
            return {tracked: provided[tracked] for tracked in wrt}
        if len(wrt) == 1 and isinstance(provided, np.ndarray):
            return {wrt[0]: provided}
        if isinstance(provided, (list, tuple)) and len(provided) == len(wrt) \
                and all(isinstance(item, np.ndarray) for item in provided):
            return {tracked: item for tracked, item in zip(wrt, provided)}
        raise self._error(
            "`tangent=` takes one direction array (a single tracked argument), a "
            "mapping `wrt` name -> direction, or one direction per tracked argument in "
            "`wrt` order")

    def _resolve_out_array(self, values:Dict[str, object])->Optional[str]:
        """The array the tangent flows into: the body's returned name, or `out=`."""
        if self.out == RETURN_SPELL:
            scan:BodyScan = _scan_of(self.kernel)
            returns:List[ast.Return] = scan.returns()
            if not returns:
                raise self._error("the body returns nothing, and no `out=` was named: "
                                  "spell `out=\"name\"` for the parameter that carries "
                                  "the result")
            names:Set[str] = set()
            for ret in returns:
                if not isinstance(ret.value, ast.Name):
                    raise self._error("the body's `return` carries an expression; the "
                                      "tangent flows into one named result -- spell "
                                      "`out=\"name\"`")
                names.add(ret.value.id)
            if len(names) != 1:
                raise self._error("the body's returns do not agree on one named result; "
                                  "spell `out=\"name\"`")
            out_name:str = names.pop()
            if out_name in self._params() and isinstance(values.get(out_name),
                                                         np.ndarray):
                return out_name
            return None               # a scalar result: the call returns its tangent value
        if not isinstance(self.out, str) or self.out not in self._params():
            raise self._error(f"`out={self.out!r}` does not name a parameter of this "
                              f"kernel")
        return self.out

    def _build(self, wrt:List[str], tangents:Dict[str, str],
               out_name:Optional[str], out_tangent:Optional[str])->object:
        return _build_jvp_kernel(self.kernel, wrt, tangents, out_name, out_tangent)


