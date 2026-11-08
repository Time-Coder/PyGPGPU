from __future__ import annotations
import ast
from typing import (Optional)
import numpy as np
from ..structs import dtype_for as structs_dtype_for
from .notation import (
    Translated,
)
from .receivers import (
    structs_name,
)
from .vectors import (
    is_vec,
)
from .predicates import (
    is_float,
    is_ptr,
)


class _QuaternionsMixin:

    def _emit_quaternion_binop(self, operator:str, left:str, lt:object, right:str, rt:object,
                               node:ast.BinOp)->Optional[Translated]:
        """A quaternion operation, as a call to one of the header's functions.

        `pygf` gives a quaternion three meanings: `q * q` composes two rotations, `q * v`
        rotates a vector, and a scalar scales it. None is a C operator on a structure, so each
        becomes a function from the fixed header and the expression becomes a call.

        A float and a double operand mix freely, and pygf's rule is that the result is the
        **double** form -- `quatf * quatd` is a `quatd`, `quatf * double3` is a `double3`,
        `quatd * float3` is a `double3`. Widening only ever goes that way, so nothing is
        rounded down; each mixed combination has its own function in the header.
        """
        from .. import quaternion  # noqa: PLC0415

        # A structure operand travels as a pointer, so a quaternion variable is `quatf*`:
        # unwrap it for the type, and dereference it in the emitted expression.
        if isinstance(lt, tuple) and is_ptr(lt) and structs_dtype_for(lt[1]) is not None:
            lt, left = lt[1], f"(*{left})"
        if isinstance(rt, tuple) and is_ptr(rt) and structs_dtype_for(rt[1]) is not None:
            rt, right = rt[1], f"(*{right})"
        left_quat:Optional[np.dtype] = quaternion.is_quaternion(lt)
        right_quat:Optional[np.dtype] = quaternion.is_quaternion(rt)
        if left_quat is None and right_quat is None:
            return None
        if operator not in ("Add", "Sub", "Mult", "Div"):
            return None

        def call(name:str, dtype:np.dtype, args:str, result:object,
                 mixed:bool = False) -> Translated:
            # The types and their arithmetic both come from the fixed header; all this records
            # is which widths are in use, so the compile step can enable their blocks. A mixed
            # operation touches both, because its body widens one width to the other.
            if mixed:
                self.quaternion_needs.update(quaternion.mixed_needs())
            else:
                self._require_quaternion_header(dtype)
            return f"{name}({args})", result

        def widen(left_dtype:np.dtype, right_dtype:np.dtype, operation:str) -> str:
            """The header function for an operation on these two operand types.

            The same width is `<type>_<operation>`; a mixed pair is
            `<left type>_<operation>_<right type>`, which is how the header spells every mixed
            combination. Deriving the name keeps the two in step without a table.
            """
            left_name:str = structs_name(left_dtype)
            if left_name == structs_name(right_dtype):
                return f"{left_name}_{operation}"
            return f"{left_name}_{operation}_{structs_name(right_dtype)}"

        if operator == "Mult":
            if left_quat is not None and right_quat is not None:
                name:str = widen(left_quat, right_quat, "mul")
                # A mixed pair answers the double form, which is a real structure the header
                # declares, so it is registered under its fixed name like any other.
                result:object = (lt if structs_name(left_quat) == structs_name(right_quat)
                              else structs_name(quaternion.double_dtype()))
                return call(name, quaternion.wider(left_quat, right_quat),
                            f"{left}, {right}", result,
                            mixed=structs_name(left_quat) != structs_name(right_quat))
            if left_quat is not None and is_vec(rt):
                # A float quaternion rotating a double vector answers a double vector, and a
                # double one rotating a float vector likewise; only the matching pair is the
                # plain `_rotate`.
                same:bool = quaternion.matches_vector(left_quat, rt)
                name = (f"{structs_name(left_quat)}_rotate" if same
                        else f"{structs_name(left_quat)}_rotate_{rt}")
                result_vec:object = rt if same else (
                    "double3" if rt.startswith("float") else rt)
                # A widened rotation needs the double block too, since it answers a double3.
                return call(name, left_quat, f"{left}, {right}", result_vec, mixed=not same)
            if left_quat is not None and is_float(rt):
                return call(f"{structs_name(left_quat)}_scale", left_quat,
                            f"{left}, {right}", lt)
            if right_quat is not None and is_float(lt):
                # `2.0 * q` multiplies the same way, and the scalar comes first in C.
                return call(f"{structs_name(right_quat)}_scale", right_quat,
                            f"{right}, {left}", rt)
            return None

        if operator == "Div" and left_quat is not None and is_float(rt):
            return call(f"{structs_name(left_quat)}_scale", left_quat,
                        f"{left}, (1.0f / ({right}))", lt)

        if operator in ("Add", "Sub"):
            # Component-wise, which is what the base class would do.
            if left_quat is None or right_quat is None:
                return None
            suffix:str = "add" if operator == "Add" else "sub"
            name = widen(left_quat, right_quat, suffix)
            result = (lt if structs_name(left_quat) == structs_name(right_quat)
                      else structs_name(quaternion.double_dtype()))
            return call(name, quaternion.wider(left_quat, right_quat), f"{left}, {right}",
                        result, mixed=structs_name(left_quat) != structs_name(right_quat))

        return None


    def _require_quaternion_header(self, dtype:np.dtype)->None:
        """Record that this source uses a quaternion of this width.

        One header serves both widths; what the compile step needs to know is which blocks to
        enable, because a `double` block does not compile on a device without fp64.
        """
        from .. import quaternion  # noqa: PLC0415

        need:Optional[str] = quaternion.needs(dtype)
        if need is not None:
            self.quaternion_needs.add(need)


    def _emit_quaternion_call(self, node:ast.Call)->Optional[Translated]:
        """`conjugate(q)` as the generated device function for the same meaning.

        `pygf.funcs.conjugate` is an ordinary python function, so without this it was
        translated as a device *helper* -- and its body builds `m.__class__(...)`, naming a
        class a kernel has no meaning for. The quaternion meaning is what the call is after,
        so it maps onto the generated `{type}_conjugate` instead.
        """
        from .. import quaternion  # noqa: PLC0415

        def called_name(expr:ast.expr)->Optional[str]:
            if isinstance(expr, ast.Name):
                return expr.id
            if isinstance(expr, ast.Attribute):
                return expr.attr
            return None

        if called_name(node.func) != "conjugate" or len(node.args) != 1:
            return None
        if isinstance(node.func, ast.Name):
            callee:object = self.resolve_global(node.func.id)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            owner:object = self.resolve_global(node.func.value.id)
            callee = getattr(owner, node.func.attr, None)
        else:
            return None
        import pygf.funcs as pygf_funcs  # noqa: PLC0415

        if callee is not pygf_funcs.conjugate:
            return None
        argument:Translated = self._visit_expression(node.args[0])
        # The argument may travel as a pointer: unwrap the type, dereference the value.
        arg_code:str = argument[0]
        arg_type:object = argument[1]
        if isinstance(arg_type, tuple) and is_ptr(arg_type):
            arg_type, arg_code = arg_type[1], f"(*{arg_code})"
        dtype:Optional[np.dtype] = quaternion.is_quaternion(arg_type)
        if dtype is None:
            # `conjugate` serves both meanings: a quaternion flips its axis, a complex value
            # its imaginary part. pygf's own `conjugate` accepts either, so the device has to
            # as well -- otherwise one kernel would mean two things per type.
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(arg_type)
            if complex_dtype is not None:
                kind:str = self._complex_side(arg_type, complex_dtype)
                self._require_complex(complex_dtype)
                return f"{kind}_conjugate({arg_code})", kind
            self._error(
                f"conjugate() needs a quaternion or a complex value, got {arg_type}", node)
        type_name:str = structs_name(dtype)
        self._require_quaternion_header(dtype)
        return f"{type_name}_conjugate({arg_code})", type_name
