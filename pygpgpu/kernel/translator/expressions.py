from __future__ import annotations
import ast
import enum
from typing import (Any, Dict, List, Optional, Tuple)
import numpy as np
from ..device_builtins import (
    INDEX_VECTOR_BUILTINS,
    INDEX_VECTOR_OPENCL,
)
from .notation import (
    DeviceType,
    HostValueEntry,
    PTR,
    Translated,
    device_name,
)
from .receivers import (
    _enum_member_value,
)
from .predicates import (
    promote,
)


class _ExpressionsMixin:

    def _visit_expression(self, node:ast.expr)->Translated:
        if isinstance(node, ast.Constant):
            return self._visit_constant(node)
        if isinstance(node, ast.Name):
            return self._visit_name(node)
        if isinstance(node, ast.BinOp):
            return self._visit_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._visit_unaryop(node)
        if isinstance(node, ast.Compare):
            return self._visit_compare(node)
        if isinstance(node, ast.BoolOp):
            # Each operand is a *condition* in python (`and`/`or` read their operands as
            # bools), so a complex operand reaches the truthiness rule instead of arriving
            # at the compiler as a struct the operator cannot take.
            parts:List[str] = [self._condition_code(v) for v in node.values]
            symbol:str = ("&&" if isinstance(node.op, ast.And) else "||")
            joined:str = f" {symbol} ".join(parts)
            return f"({joined})", "bool"
        if isinstance(node, ast.IfExp):
            cond:str = self._condition_code(node.test)
            then_code, tt = self._visit_expression(node.body)
            else_code, et = self._visit_expression(node.orelse)
            if tt != et:
                unified:object = promote(tt, et)
                return f"({cond} ? ({unified})({then_code}) : ({unified})({else_code}))", unified
            return f"({cond} ? {then_code} : {else_code})", tt
        if isinstance(node, ast.Call):
            return self._visit_call(node)
        if isinstance(node, ast.Subscript):
            return self._visit_subscript(node)
        if isinstance(node, ast.Attribute):
            return self._visit_attribute(node)
        self._error(f"unsupported expression: {node.__class__.__name__}", node)


    def _visit_enum_constant(self, entry:Tuple[Any, Any], node:ast.AST)->Translated:
        """Emit an enum member as the constant it stands for.

        An `IntEnum` member's value is an int and an `Enum`'s may be anything; only the
        numeric ones have a device meaning, so a member whose value is not a number is
        refused with the value's type named -- there is nothing to emit for it.
        """
        payload:object = entry[0]
        member:object = entry[1]
        name:str = f"{type(member).__name__}.{getattr(member, 'name', '?')}"
        if isinstance(payload, enum.Enum):
            payload = payload.value
        if isinstance(payload, bool):
            return ("true" if payload else "false"), "bool"
        if isinstance(payload, int):
            if payload == 0:
                return "0", "int"
            # An enumerator is spelled in the C source, so a negative one keeps its sign.
            return str(payload), "int"
        if isinstance(payload, float):
            return self._visit_constant(ast.Constant(value=payload))
        self._error(
            f"{name} is an enum member whose value is a {type(payload).__name__}, which has "
            f"no device equivalent: give the member a numeric value (an int or a float), or "
            f"read the value on the host and pass it as a scalar argument", node)


    def _visit_constant(self, node:ast.Constant)->Translated:
        value:object = node.value
        if isinstance(value, bool):
            return ("true" if value else "false"), "bool"
        if isinstance(value, int):
            return str(value), "int"
        if isinstance(value, float):
            # A python float literal has no width of its own, so it takes the one the
            # `float` mapping chooses -- the same rule a python complex literal follows
            # (`1 + 2j` is a c128 under `float=np.float64`), and the same rule a python
            # float *argument* follows. Spelling every literal with the `f` suffix cut a
            # 17-digit constant down to float32 inside a double kernel.
            if self.type_map.get("float", "float") == "double":
                return self._real_literal(value, "double"), "double"
            text:str = repr(value)
            if not ("." in text or "e" in text or "E" in text or "inf" in text or "nan" in text):
                text += ".0"
            if not text.endswith(("f", "F", "y")) and "inf" not in text:
                text += "f"
            return text, "float"
        if isinstance(value, complex):
            # A complex literal is written with its two parts, because C has no complex
            # constant: the header builds the value from the reals. A python complex has no
            # width of its own, so it takes the one the `float` mapping chooses -- the same
            # rule a python float follows -- which is why `1 + 2j` meets a complex64 operand
            # as a complex64 while `@kernel(float=np.float64)` makes it a complex128.
            kind:str = self._python_complex_kind()
            return self._emit_complex_literal(value.imag, value.real, kind), kind
        self._error(f"unsupported constant: {value!r}", node)


    def _visit_name(self, node:ast.Name)->Translated:
        host:Optional[HostValueEntry] = self.host_values.get(node.id)
        if host is not None:
            # The host created this object; on the device it is just the parameter that
            # carries it (see _host_created_assignment).
            return device_name(node.id), host["type"]
        host_scalar:Optional[DeviceType] = self.host_scalars.get(node.id)
        if host_scalar is not None:
            # A host-region statement bound this value; the device reads the parameter
            # that carries it (see _bind_host_statement).
            return device_name(node.id), host_scalar
        outer:Optional[Dict[str, object]] = self.outer_vars.get(node.id)
        if outer is not None:
            # A writable outer variable travels as `T*`, so every use is a dereference; a
            # read-only one travels as `T` and is simply the parameter itself.
            if outer["writable"]:
                return f"(*{device_name(node.id)})", outer["type"]
            return device_name(node.id), outer["type"]
        t:DeviceType = self._lookup(node.id)
        if isinstance(t, tuple) and len(t) == 2 and t[0] == "carried":
            # One location for the whole launch. A subscript rather than (*name) for the
            # same reason an outer scalar uses one: the read/write analysis only recognizes
            # that form as a write, so the host reads the final value back.
            return f"{device_name(t[1])}[0]", "int"
        if t is not None:
            if isinstance(t, tuple) and len(t) == 5 and t[0] == "shared":
                # A shared buffer is not itself a pointer: it is one slice per group, and
                # only uf[block_id(d)] picks the caller's out (see _visit_shared_slice).
                self._error(
                    f"'{node.id}' is a zeros_per_group() buffer, which is one slice per work group: "
                    f"index it with the group first, as '{node.id}[blockIdx]'", node)
            if isinstance(t, tuple) and len(t) == 4 and t[0] == "bodyarr":
                # A private array created in the body is indexed like any array; its extents
                # are constants (see _shape_var).
                return device_name(node.id), (PTR, t[3])
            return device_name(node.id), t
        # CUDA's index vectors are values, not calls (`buf[blockIdx][j]`), so the *name*
        # resolves to the vector the backend builds from its own launch terms.
        if node.id in INDEX_VECTOR_BUILTINS:
            table:Dict[str, Tuple[str, str]] = (
                INDEX_VECTOR_OPENCL if self._opencl_syntax else INDEX_VECTOR_BUILTINS)
            vector_type, code = table[node.id]
            return code, vector_type
        value:object = self.resolve_global(node.id)
        enum_value:Optional[Tuple[Any, Any]] = _enum_member_value(value)
        if enum_value is not None:
            # An enum member is a named constant, so the device reads its value. Python has
            # no device enum type to name here -- the number is what every operator and
            # every function that accepts it already wants.
            return self._visit_enum_constant(enum_value, node)
        if isinstance(value, bool):
            return self._visit_constant(ast.Constant(value=value))
        if isinstance(value, (int, float)) and not isinstance(value, np.ndarray):
            return self._visit_constant(ast.Constant(value=value))
        self._error(self._unknown_name_message(node), node)


    def _unknown_name_message(self, node:ast.Name)->str:
        """Why a name the body plainly assigns cannot be read here.

        The common way to reach this is an assignment that happens inside a branch: the name
        is a local of the function, but C has no declaration for it until the statement that
        assigns it runs, so reading it after the branch fails. Python says the same thing
        differently -- it raises UnboundLocalError, and only when the branch did not run --
        which is why this reads better as a statement about where the first assignment is
        than as "unknown name".
        """
        name:str = node.id
        if name in self._local_names():
            return (f"'{name}' is assigned inside a branch or loop, so it has no value yet "
                    f"at this point: give it one before the branch (`{name} = 0`) so the "
                    f"branch only updates it")
        return (f"unknown name '{name}' (only parameters, local variables and int/float/bool "
                f"global constants are supported)")


    def _real_literal(self, value:float, width:str)->str:
        """A real literal of the given width, spelled so the C compiler reads it as one."""
        text:str = repr(float(value))
        if not ("." in text or "e" in text or "E" in text or "inf" in text or "nan" in text):
            text += ".0"
        if width != "double" and not text.endswith(("f", "F", "y")) and "inf" not in text:
            text += "f"
        return text


    def _condition_code(self, node:ast.expr)->str:
        """A condition as C reads it: `(code)` for what is already a scalar test.

        A complex value is truthy in python when either part is non-zero, so a condition
        holding one (`if z:`, `while z and n < max:`) is rewritten as the comparison the
        record itself cannot express -- the plain `if (z)` reached the compiler and was
        refused with "expression must have bool type". A zero complex value means
        `z.re == 0 && z.im == 0`, which is the negation of the header's own inequality
        against the zero the constructor builds.
        """
        code, t = self._visit_expression(node)
        dtype:Optional[np.dtype] = self._complex_dtype_of(t)
        if dtype is None:
            return f"({code})"
        self._require_complex(dtype)
        kind:str = self._complex_side(t, dtype)
        return (f"({kind}_equal({code}, {kind}_from_real(0)) == 0)")
