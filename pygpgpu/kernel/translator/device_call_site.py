# How ONE call site binds the shape extents a helper's signature declares.
from __future__ import annotations
import ast
from typing import Callable, List, Tuple


class _DeviceCallSiteMixin:

    def _site_shape_codes(self, raw_func:Callable, shape_params:List[Tuple[str, int]],
                          node:ast.Call, receiver_offset:int)->List[str]:
        """The shape-variable arguments one call site passes for a helper's extents.

        The helper's signature declares each extent under the helper's own name
        (`const int _grid_shape1`, see `render_device`); the argument this site passes
        is the extent of the array *this site* bound to that parameter, spelled in the
        caller's own terms (`_self_grid_shape1` when the array is the receiver's `grid`
        attribute). The registration rides on `_shape_var`/`_tail_var`, so the caller's
        `shapes_used` gains the caller-side name -- the one the launch can resolve
        against the caller's parameter list -- and the generated call hands the helper
        the caller's variable.
        """
        codes:List[str] = []
        for array_name, dim in shape_params:
            bound:str = self._bound_shape_name(raw_func, array_name, node, receiver_offset)
            codes.append(self._shape_var(bound, dim) if dim >= 0
                         else self._tail_var(bound, -dim - 2))
        return codes


    def _bound_shape_name(self, raw_func:Callable, array_name:str, node:ast.Call,
                          receiver_offset:int)->str:
        """The caller-side spelling of the array one helper shape extent names.

        The callee's `shapes_used` names are the callee's own: a parameter it was
        handed (`grid`), its own receiver attribute (`_self_grid`), or an outer name.
        The last two are already caller terms -- the attributes are forwarded under
        the caller's own names and an outer variable travels by name -- so only a
        *parameter* needs the call's binding: the argument this call passed in that
        parameter's position names the array on the caller's side. A bare name is
        rooted through the pointer-alias chain (`src = self.grid` makes `src` read as
        `_self_grid`), and a `self.grid` argument names the attribute slot directly.
        An argument this mapping cannot read -- an expression, a slice, a call's
        result -- leaves the callee's own name, which is what the shape variable was
        called before this mapping existed.
        """
        code:object = getattr(raw_func, "__code__", None)
        if code is not None:
            params:Tuple[str, ...] = code.co_varnames[:code.co_argcount]
            if array_name in params:
                position:int = params.index(array_name) - receiver_offset
                if 0 <= position < len(node.args):
                    argument:ast.expr = node.args[position]
                    if isinstance(argument, ast.Name):
                        return self._array_pointer_root(argument.id)
                    if (self.self_param and isinstance(argument, ast.Attribute)
                            and isinstance(argument.value, ast.Name)
                            and argument.value.id == self.self_param):
                        return f"_self_{argument.attr}"
        return array_name
