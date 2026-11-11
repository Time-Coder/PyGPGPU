from __future__ import annotations
import ast
from typing import (Any, Dict, List, Optional, Tuple)
from .type_map import (
    _check_integer_fits,
)
from .notation import (
    Translated,
)
from .receivers import (
    MAGIC_METHODS,
)
from .vectors import (
    is_vec,
)
from .predicates import (
    is_float,
    is_int,
    promote,
)


class _OperatorsMixin:

    def _visit_binop(self, node:ast.BinOp)->Translated:
        left, lt = self._visit_expression(node.left)
        right, rt = self._visit_expression(node.right)
        name:str = node.op.__class__.__name__

        # A quaternion is a structure too, but its arithmetic *does* have a device meaning
        # (pygf defines it), so it is handled by generated device functions before the
        # general structure refusal below.
        quat_result:Optional[Translated] = self._emit_quaternion_binop(
            name, left, lt, right, rt, node)
        if quat_result is not None:
            return quat_result

        # A complex value is a structure too, and its arithmetic likewise has a device
        # meaning -- numpy's own, which the header implements -- so it is answered here,
        # before the general structure refusal below.
        complex_result:Optional[Translated] = self._emit_complex_binop(
            name, left, lt, right, rt, node)
        if complex_result is not None:
            return complex_result

        # A class that defines the operator's magic method gives it a meaning, and the
        # translation keeps that meaning: `a + b` becomes `Class___add__(&a, &b)`, the same
        # shape `ray.at(t)` takes. Only a method written in the class body counts, so an
        # inherited identity comparison cannot silently answer for a structure.
        magic:Optional[str] = MAGIC_METHODS.get(name)
        if magic is not None:
            call:Optional[Translated] = self._emit_magic_binop(
                magic, left, lt, right, rt, node)
            if call is not None:
                return call

        # A structure value has no arithmetic: none of `+ - * / % ** < <= > >= == !=` has a
        # device meaning for an aggregate, and emitting one produced either a C++ operator
        # error or -- worse for `Mult` -- a per-component scalar multiply that silently
        # computed something the language never defined. Every other structure is refused
        # here, before the vector dispatch, so the message can say what to do instead.
        if self._is_structure_operand(lt) or self._is_structure_operand(rt):
            self._error(self._structure_operator_message(name, lt, rt, node), node)

        if is_vec(lt) or is_vec(rt):
            symbol:str = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/",
                          "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
                          "Eq": "==", "NotEq": "!="}.get(name)
            if symbol is None:
                self._error(f"operator {name} is not supported on vector types; operate per-component", node)

            if is_vec(lt) and is_vec(rt):
                if lt != rt:
                    self._error(f"vector operands must have the same type, got {lt} and {rt}", node)
                if self._opencl_syntax:
                    result_t:object = (lt if name in ("Add", "Sub", "Mult", "Div") else self._get_int_vec_type(lt))
                    return f"({left} {symbol} {right})", result_t
                if name in ("Add", "Sub", "Mult", "Div"):
                    return self._vec_componentwise(symbol, left, lt, right=right), lt
                return self._vec_componentwise_compare(symbol, left, lt, right), self._get_int_vec_type(lt)

            # vector with scalar: broadcast per component
            vec, vec_side_left = (lt, True) if is_vec(lt) else (rt, False)
            if self._opencl_syntax:
                code:str = f"({left} {symbol} {right})" if vec_side_left else f"({right} {symbol} {left})"
                return code, vec
            if vec_side_left:
                return self._vec_componentwise(symbol, left, vec, scalar=right), vec
            return self._vec_componentwise(symbol, right, vec, scalar=left), vec

        if name == "Div":
            t:object = promote(lt, rt) if not (is_int(lt) and is_int(rt)) else "float"
            if is_int(lt) and is_int(rt):
                return f"((({t})({left})) / (({t})({right})))", t
            return f"({left} / {right})", t

        if name == "FloorDiv":
            if is_int(lt) and is_int(rt):
                # C's `/` truncates and python's `//` floors, so the two disagree for a
                # negative pair. The helper applies the correction (see
                # device_builtins.PYTHON_FLOOR_SOURCES); for an unsigned width it is
                # empty and the plain operator already agrees.
                return self._emit_floored(("floor_div", "floor_mod"), "floor_div",
                                          left, promote(lt, rt), right, rt, node)
            t:object = promote(lt, rt)
            return f"(({t})floor(({t})({left}) / ({t})({right})))", t

        if name == "Pow":
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and node.right.value == 2:
                return f"({left} * {left})", lt
            t:object = promote(lt, rt)
            if is_int(t):
                t = "float"
            return f"(({t})pow(({t})({left}), ({t})({right})))", t

        if name in ("LShift", "RShift", "BitOr", "BitAnd", "BitXor"):
            if not (is_int(lt) and is_int(rt)):
                self._error(f"operator {name} requires integer operands", node)
            symbol:str = {"LShift": "<<", "RShift": ">>", "BitOr": "|", "BitAnd": "&", "BitXor": "^"}[name]
            return f"({left} {symbol} {right})", promote(lt, rt)

        if name == "Mod":
            t:object = promote(lt, rt)
            if is_int(t):
                # Same discrepancy as `//`: C's `%` takes the dividend's sign and python's
                # takes the divisor's.
                return self._emit_floored(("floor_div", "floor_mod"), "floor_mod",
                                          left, t, right, rt, node)
            if is_float(t):
                # C's `%` is an integer operator, so the old emission was a bare `fmod`
                # call -- whose result carries the *dividend's* sign, where python's
                # remainder carries the *divisor's* (`-1.0 % 4.0` is 3.0 in python and
                # -1.0 from fmod). The helper is CPython's own float remainder: fmod's
                # result, plus the divisor once when the two signs disagree.
                return self._emit_floored(("floor_div", "floor_mod"), "float_mod",
                                          left, t, right, rt, node)
            return f"fmod(({t})({left}), ({t})({right}))", t

        symbol:str = {"Add": "+", "Sub": "-", "Mult": "*"}.get(name)
        if symbol is None:
            self._error(f"unsupported operator: {name}", node)
        result_t:object = promote(lt, rt)
        if is_int(lt) and is_int(rt):
            # Python's int is arbitrary precision and the device's is the mapped width, so a
            # product of two literal ints used to wrap silently: `5000 * 1000000` answered
            # 705032704 with a compiler warning nobody piped up. When both sides are literal
            # arithmetic the host can compute the exact value and run the same width check a
            # scalar *argument* gets (`_check_integer_fits`); a literal that does not fit is
            # refused with the numpy spelling that carries it, not wrapped.
            folded:Optional[int] = self._folded_int_value(node)
            if folded is not None:
                _check_integer_fits(folded, result_t if isinstance(result_t, str) else "int")
        return f"({left} {symbol} {right})", result_t


    def _folded_int_value(self, node:ast.BinOp)->Optional[int]:
        """The exact value of a literal int BinOp, or None when it is not foldable.

        A "literal" here is an expression built only from int constants, the arithmetic
        operators python folds with (`+ - * / // % **`), and unary signs --
        `5000000 * 1000000`, `2 ** 40`, `-7 // 2`. A name (a parameter or a local) is *not*
        folded: its value is a launch-time fact, and the mapping `int=np.int64` is the
        escape hatch for a runtime product that needs the wider device type.
        """
        operations:Dict[type, object] = {
            ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b, ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b, ast.Pow: lambda a, b: a ** b,
        }

        def evaluate(expr:ast.expr)->Optional[int]:
            if isinstance(expr, ast.Constant) and isinstance(expr.value, int) \
                    and not isinstance(expr.value, bool):
                return int(expr.value)
            if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
                inner:Optional[int] = evaluate(expr.operand)
                if inner is None:
                    return None
                return -inner if isinstance(expr.op, ast.USub) else inner
            if isinstance(expr, ast.BinOp) and type(expr.op) in operations:
                left:Optional[int] = evaluate(expr.left)
                right:Optional[int] = evaluate(expr.right)
                if left is None or right is None:
                    return None
                try:
                    return int(operations[type(expr.op)](left, right))
                except (ZeroDivisionError, ValueError, OverflowError):
                    return None
            return None

        return evaluate(node)


    def _visit_unaryop(self, node:ast.UnaryOp)->Translated:
        code, t = self._visit_expression(node.operand)
        name:str = node.op.__class__.__name__
        # A class that defines `__neg__` / `__pos__` / `__invert__` gives the unary operator
        # a meaning; without this, `-a` on a structure emitted `(-a)` and the compiler
        # rejected it as arithmetic on an aggregate.
        magic:Optional[str] = MAGIC_METHODS.get(name)
        if magic is not None:
            owner:Optional[Tuple[type, Any]] = self._magic_owner(t, magic)
            if owner is not None:
                cls, method = owner
                return self._emit_device_call(
                    magic, method, [(code, t)], node,
                    self_param=(method.__code__.co_varnames[0]
                                if getattr(method, "__code__", None) else "self"),
                    receiver_as_argument=True)
        if name == "USub":
            if is_vec(t) and self.backend == "cuda":
                # CUDA has no unary minus on a vector type: the negation is per component.
                return self._vec_componentwise("-", code, t), t
            if self._complex_dtype_of(t) is not None:
                # A complex value is a structure, so `-z` is the header's own negation.
                self._require_complex(self._complex_dtype_of(t))
                kind:str = self._complex_side(t, self._complex_dtype_of(t))
                return f"{kind}_neg({code})", kind
            return f"(-{code})", t
        if name == "UAdd":
            if self._complex_dtype_of(t) is not None:
                # `+z` is z: python answers the value unchanged, and so does a C++ copy.
                return code, self._complex_side(t, self._complex_dtype_of(t))
            return f"(+{code})", t
        if name == "Invert":
            if not is_int(t):
                self._error("~ requires an integer operand", node)
            return f"(~{code})", t
        if name == "Not":
            # The operand is parenthesised, or `not a == b` reaches C as `!a == b` -- a
            # comparison against the negated left operand, which is not what python means
            # (`not (a == b)`), and the OpenCL compiler says so with "logical not is only
            # applied to the left hand side of this comparison".
            return f"(!({code}))", "bool"
        self._error(f"unsupported unary operator: {name}", node)


    def _visit_compare(self, node:ast.Compare)->Translated:
        symbols:Dict[str, str] = {"Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
                                  "Eq": "==", "NotEq": "!=", "In": "in", "NotIn": "not in"}

        left, lt = self._visit_expression(node.left)
        if is_vec(lt):
            # vector comparison (opencl natively returns an int vector; cuda generates per-component), same semantics as genVec._compare_op
            if len(node.ops) != 1:
                self._error("chained comparison is not supported on vector types", node)
            name:str = node.ops[0].__class__.__name__
            if name not in symbols:
                self._error(f"unsupported comparison: {name}", node)
            right, rt = self._visit_expression(node.comparators[0])
            if is_vec(rt):
                if rt != lt:
                    self._error(f"vector operands must have the same type, got {lt} and {rt}", node)
                return self._vec_componentwise_compare(symbols[name], left, lt, right), self._get_int_vec_type(lt)
            self._error("vector comparison requires vector operands on both sides", node)

        parts:List[str] = []
        for op, comparator in zip(node.ops, node.comparators):
            name:str = op.__class__.__name__
            if name not in symbols:
                self._error(f"unsupported comparison: {name}", op)
            right, rt = self._visit_expression(comparator)
            # A class that defines `__eq__` / `__lt__` and its siblings gives the comparison
            # a meaning, and the translation keeps it as a device call.
            magic:Optional[str] = MAGIC_METHODS.get(name)
            if magic is not None:
                call:Optional[Translated] = self._emit_magic_compare(
                    magic, left, lt, right, rt, op)
                if call is not None:
                    # A chained comparison cannot be a single call: `a < b < c` means
                    # `a < b and b < c`, which the loop already joins with `&&`.
                    if len(node.ops) != 1:
                        self._error(
                            "chained comparison is not supported on a class that defines "
                            "its own comparison; write the two comparisons separately", op)
                    return call
            # A complex value compares with `==` and `!=` (numpy compares the two parts), and
            # nothing else: python has no ordering for a complex number, so numpy and the
            # device raise the same TypeError rather than inventing one.
            if self._complex_dtype_of(lt) is not None or self._complex_dtype_of(rt) is not None:
                if name not in ("Eq", "NotEq"):
                    self._error(
                        f"complex numbers have no '{symbols[name]}' comparison: python orders"
                        f" no complex value, so only == and != are defined. Compare the"
                        f" magnitude (abs) or a component (.real / .imag) instead", op)
                call:Optional[Translated] = self._emit_complex_compare(name, left, lt, right, rt)
                if call is None:
                    self._error(
                        f"cannot compare {lt} with {rt}: a complex value compares with a"
                        f" complex number, or with a real one it is widened to", op)
                parts.append(call[0])
                left = right
                continue
            # A structure value has no comparison either: C rejects `s == s` outright, so it
            # is refused here with the same explanation the arithmetic gets.
            if self._is_structure_operand(lt) or self._is_structure_operand(rt):
                self._error(self._structure_operator_message(name, lt, rt, node), op)
            parts.append(f"{left} {symbols[name]} {right}")
            left = right
        # A comparison is not wrapped here: an `if` adds the one pair C's syntax needs, and a
        # second pair made it emit `if (((a == b)))` -- the middle one, which the OpenCL
        # compiler reports as "equality comparison with extraneous parentheses" (and points
        # at `=` as a "fix", which would silently change the meaning). An operand that needs
        # grouping of its own -- a nested comparison, an arithmetic expression -- already
        # carries parentheses from its own visitor. A chained comparison joins its parts with
        # `&&`, and the `&&` expression gets the pair that groups it.
        if len(parts) > 1:
            return f"({' && '.join(parts)})", "bool"
        return parts[0], "bool"
