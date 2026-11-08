from __future__ import annotations
import ast
from typing import (List, Optional, Tuple)
import numpy as np
from .. import complex as complex_support
from .type_map import (
    python_complex_type,
)
from .notation import (
    Translated,
)
from .vectors import (
    is_vec,
)
from .predicates import (
    is_float,
    is_image,
    is_int,
    is_ptr,
)


class _ComplexValuesMixin:

    def _complex_store(self, target_t:object, value_t:object, code:str)->str:
        """`code` as a value the target type can hold, for a complex assignment.

        C has no complex conversion operator, so a store between two different widths --
        `out64[i] = z` where `z` is a complex128, or a return slot of the other width -- would
        emit `c64 = c128` and fail to compile. The header's widening and narrowing functions
        are the conversion, and they are the *same* ones a cast uses, so a store and an
        explicit `np.complex64(z)` mean the same thing.

        A real source assigned into a complex target is lifted to the complex value with a
        zero imaginary part, which is numpy's rule (`out64[i] = 2.0` is `2 + 0i`).
        """
        target_complex:Optional[np.dtype] = self._complex_dtype_of(target_t)
        value_complex:Optional[np.dtype] = self._complex_dtype_of(value_t)
        if target_complex is None:
            if value_complex is not None:
                self._error(
                    f"cannot assign a complex value ({value_t}) into a {target_t}: a complex"
                    f" number has no real counterpart, so write `.real` or `.imag` if that"
                    f" is what you meant")
            return code
        kind:str = self._complex_side(target_t, target_complex)
        self._require_complex(target_complex)
        if value_complex is None:
            if is_ptr(value_t) or is_vec(value_t) or is_image(value_t):
                return code
            element:str = "double" if kind == "c128" else "float"
            return f"{kind}_from_real(({element})({code}))"
        value_kind:str = self._complex_side(value_t, value_complex)
        return self._convert_complex(code, value_kind, kind)


    def _convert_complex(self, code:str, from_kind:str, to_kind:str)->str:
        """`code` as a value of the other complex width, through the header's conversion.

        C has no complex conversion operator, so the header's widening and narrowing functions
        are the conversion. They are the *same* ones a cast uses, so a store, an explicit
        `np.complex64(z)` and a python complex reaching the mapped width all mean one thing.
        """
        if from_kind == to_kind:
            return code
        self.complex_needs.update(complex_support.mixed_needs())
        converter:str = (f"{from_kind}_widen_{to_kind}" if to_kind == "c128"
                         else f"{from_kind}_narrow_{to_kind}")
        return f"{converter}({code})"


    def _python_complex_kind(self)->str:
        """The header's spelling of the width a python `complex` travels as in this kernel.

        The one decision point on this side of the translation: the mapping's `float` entry
        chooses it through `python_complex_type`, so a body literal and `complex(re, im)`
        cannot disagree with the `c64`/`c128` a python complex *argument* is declared as.
        """
        return python_complex_type(self.type_map.get("float", "float"))


    def _emit_complex_literal(self, imaginary:float, real:float, kind:str)->str:
        """One complex constant, as the header's constructor over two real literals."""
        element:str = "double" if kind == "c128" else "float"
        self._require_complex(np.dtype(np.complex128 if kind == "c128" else np.complex64))
        return (f"{kind}_make({self._real_literal(real, element)},"
                f" {self._real_literal(imaginary, element)})")


    def _require_complex(self, dtype:object)->None:
        """Record that this source uses a complex width, and enable its block."""
        need:Optional[str] = complex_support.needs(dtype)
        self.uses_complex = True
        if need is not None:
            self.complex_needs.add(need)


    def _complex_dtype_of(self, t:object)->Optional[np.dtype]:
        """The complex dtype behind a translated type name, or None."""
        base:object = t[1] if isinstance(t, tuple) and is_ptr(t) else t
        if not isinstance(base, str):
            return None
        return complex_support.dtype_for_type_name(base)


    def _complex_operand(self, code:str, t:object)->Tuple[str, Optional[np.dtype]]:
        """A complex operand as a value: a pointer is dereferenced first."""
        dtype:Optional[np.dtype] = self._complex_dtype_of(t)
        if dtype is not None and isinstance(t, tuple) and is_ptr(t):
            return f"(*{code})", dtype
        return code, dtype


    def _complex_side(self, t:object, dtype:Optional[np.dtype])->str:
        """One operand's width, in the header's spelling: ``c64``/``c128``/``float``/``double``."""
        if dtype is not None:
            return complex_support.complex_type_name(dtype) or "c64"
        base:object = t[1] if isinstance(t, tuple) and is_ptr(t) else t
        return "double" if base == "double" else "float"


    def _emit_complex_binop(self, operator:str, left:str, lt:object, right:str, rt:object,
                            node:ast.BinOp)->Optional[Translated]:
        """A complex operation, as a call to one of the header's functions.

        A complex value is a structure on the device, so none of `+ - * / **` is a C operator:
        each becomes a function from the fixed header, named after the operand widths. The
        arithmetic itself is numpy's, which is also what the interpreter computes.

        A float and a double operand mix freely and the result is the **double** form, the same
        rule the quaternion arithmetic follows: `c64 + c128` is a `c128`, `c64 * 2.0` (a python
        float, which is a double) is a `c128`, while `c64 * np.float32(2)` stays a `c64`.
        Widening only ever goes that way, so nothing is rounded down.
        """
        left_code, left_dtype = self._complex_operand(left, lt)
        right_code, right_dtype = self._complex_operand(right, rt)
        if left_dtype is None and right_dtype is None:
            return None
        if operator not in ("Add", "Sub", "Mult", "Div", "Pow"):
            return None

        left_side:str = self._complex_side(lt, left_dtype)
        right_side:str = self._complex_side(rt, right_dtype)
        widened:bool = "c128" in (left_side, right_side) or "double" in (left_side, right_side)
        kind:str = "c128" if widened else "c64"
        suffix:str = {"Add": "add", "Sub": "sub", "Mult": "mul", "Div": "div",
                      "Pow": "pow"}[operator]

        # A mixed operation's body widens one width to the other, so its source needs both
        # blocks; a same-width one needs only its own.
        if left_dtype is not None and right_dtype is not None \
                and left_side != right_side:
            self.complex_needs.update(complex_support.mixed_needs())
        elif (left_dtype is not None and right_dtype is not None) and widened:
            self._require_complex(np.dtype(np.complex128))
        elif not widened and (left_dtype is not None or right_dtype is not None):
            self._require_complex(np.dtype(np.complex64))
        elif widened:
            self._require_complex(np.dtype(np.complex128))
            if (left_side == "double" or right_side == "double") \
                    and (left_side == "c64" or right_side == "c64"):
                self.complex_needs.update(complex_support.mixed_needs())

        if left_dtype is not None and right_dtype is not None:
            if left_side == right_side:
                return f"{kind}_{suffix}({left_code}, {right_code})", kind
            return f"{left_side}_{suffix}_{right_side}({left_code}, {right_code})", kind

        # One complex operand and one real scalar. The scalar's own width decides whether the
        # result widens, and the header names the mixed conversion after the widths it joins.
        complex_is_left:bool = left_dtype is not None
        complex_side:str = left_side if complex_is_left else right_side
        scalar_side:str = right_side if complex_is_left else left_side
        complex_code:str = left_code if complex_is_left else right_code
        scalar_code:str = right_code if complex_is_left else left_code
        mixed_scalar:bool = scalar_side == "double"
        # The real operand is cast to the width the function declares: a literal `1` in
        # `1 + 2j` is an int in C, and the header's `float`/`double` parameter would then be
        # ambiguous (CUDA rejects it with "no suitable constructor exists").
        element:str = "double" if mixed_scalar or kind == "c128" else "float"
        typed_scalar:str = f"(({element})({scalar_code}))"
        # `kind` is the *result* width, so a c64 operand meeting a double scalar still has to
        # be widened to reach the c128 function the header declares.
        typed_complex:str = complex_code
        if mixed_scalar and complex_side == "c64":
            typed_complex = f"c64_widen_c128({complex_code})"
            self.complex_needs.update(complex_support.mixed_needs())

        def scalar_call(operation:str)->Translated:
            """`z op s`, as the header's function: the scalar is the second argument.

            Never `f"{kind}_{operation}_c128"`: the header's `_c128` suffix marks an operand
            of that width (`c64_scale_c128(c64, double)`), and because `kind` here is already
            the widened result, appending it named helpers that do not exist -- every
            complex-op-float spelling was refused with "c128_scale_c128 is undefined".
            """
            return f"{kind}_{operation}({typed_complex}, {typed_scalar})", kind

        def reversed_call(operation:str)->Translated:
            """`s op z`, whose scalar is the *first* argument the header declares."""
            return f"{kind}_{operation}({typed_scalar}, {typed_complex})", kind

        if operator == "Mult":
            # `s * z` and `z * s` are the same product: the scalar is the second argument.
            return scalar_call("scale")
        if operator == "Div":
            if complex_is_left:
                return scalar_call("scale_div")
            return reversed_call("rdiv")
        if operator in ("Add", "Sub"):
            # `z + s` and `z - s` add or subtract the real part; `s + z` is the *same* sum as
            # `z + s` (addition commutes, so it reuses the same function), while `s - z` has to
            # be its own operation -- the earlier spelling used `rsub` for both, which turned
            # the commonest constant in a body (`3 + 4j`, an `Add` over a real 3) into `3 - 4j`.
            if operator == "Add":
                if not complex_is_left:
                    scalar_code, complex_code = complex_code, scalar_code
                return scalar_call("add_real")
            if complex_is_left:
                return scalar_call("sub_real")
            return reversed_call("rsub")
        if operator == "Pow":
            if complex_is_left:
                # `z ** 2` (the Mandelbrot iteration shape) is the direct product: the
                # general power goes through exp(2 * log(z)), whose round trip loses the
                # real part entirely at c64 (`(4-4j) ** 2` answered -1.4e-06 - 32j where
                # numpy answers -32j). The real path special-cases a constant 2 into
                # `x * x` (see _visit_binop); the complex path owes the same case. An
                # integer exponent of 2 only -- python's `z ** 2.0` is a float power.
                if isinstance(node.right, ast.Constant) \
                        and isinstance(node.right.value, int) \
                        and node.right.value == 2 and not isinstance(node.right.value, bool):
                    return f"{kind}_mul({complex_code}, {complex_code})", kind
                return scalar_call("pow_real")
            return reversed_call("rpow")
        return None


    def _emit_complex_constructor(self,
                                  node:ast.Call)->Translated:
        """`complex(re, im)` -- python's own constructor for a complex value.

        Python offers three spellings and they all mean the same value, so all three are
        accepted: `complex(re, im)`, `complex(z)` (a copy, or a real widened), and
        `complex("1+2j")` is refused because a string has no device value -- the text would
        have to be parsed on the host, and a caller who knows the value can write it as a
        literal.

        The result carries the width the `float` mapping chooses for a python `complex`, which
        is the same rule `1 + 2j` follows: `c64` under the 32-bit default, `c128` when the
        kernel asks for 64-bit reals. A caller who wants the other width for one value writes
        `np.complex64(complex(re, im))` (or `np.complex128(...)`), and that cast converts it.
        """
        argv:List[ast.expr] = list(node.args)
        if len(argv) not in (1, 2):
            self._error(
                f"complex() takes 1 or 2 arguments (complex(re, im) or complex(z)), got"
                f" {len(argv)}", node)
        for argument in argv:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                # `complex("3+4j")` parses the text on the host. A device has no parser, so the
                # text would have to be translated into the value it spells -- and a caller who
                # knows the value can write it as a literal or as `complex(re, im)`.
                self._error(
                    f'complex("{argument.value}") is a string parse, which device code cannot'
                    f" do: write the value it spells instead (e.g. 3 + 4j), or pass the real"
                    f" and imaginary parts as scalars", argument)
        args:List[Translated] = [self._visit_expression(argument) for argument in argv]
        for code, t in args:
            if is_ptr(t) or is_image(t) or is_vec(t):
                self._error(
                    f"complex() components must be scalars, got {t}", node)
        kind:str = self._python_complex_kind()
        element:str = "double" if kind == "c128" else "float"
        dtype:np.dtype = np.dtype(np.complex128 if kind == "c128" else np.complex64)
        if len(args) == 1:
            code, t = args[0]
            complex_dtype:Optional[np.dtype] = self._complex_dtype_of(t)
            if complex_dtype is not None:
                # `complex(z)` is z at the width a python complex travels as here: python
                # answers its own (double) type, and the mapping stands in for that type, so a
                # complex64 operand beside `float=` on a double is widened and a complex128 one
                # under the 32-bit default is narrowed -- exactly as a python float argument is.
                self._require_complex(dtype)
                value_kind:str = self._complex_side(t, complex_dtype)
                return self._convert_complex(code, value_kind, kind), kind
            if not is_float(t) and not is_int(t):
                self._error(
                    f"complex({t}) has no device meaning: pass a real or a complex value,"
                    f" or use complex(re, im)", node)
            # A real value, widened: the imaginary part is zero.
            self._require_complex(dtype)
            return f"{kind}_from_real(({element})({code}))", kind

        (real_code, real_t), (imag_code, imag_t) = args
        for code, t in args:
            if not is_float(t) and not is_int(t):
                self._error(
                    f"complex(re, im) needs real components, got {t}", node)
        # A numpy scalar part keeps its own width, as it does everywhere else
        # (`np.float64` arguments are narrowed deliberately, but a value that *travels*
        # as a double carries its width into the result): numpy's own
        # `complex(np.float64(1.0), np.float64(2.0))` answers a complex128. A *python*
        # float part has no width of its own and follows the mapping, which is the rule
        # `1 + 2j` already uses -- the mapping is what chose `kind` above.
        parts_are_double:bool = real_t == "double" or imag_t == "double"
        if parts_are_double:
            kind = "c128"
        element = "double" if kind == "c128" else "float"
        dtype = np.dtype(np.complex128 if kind == "c128" else np.complex64)
        self._require_complex(dtype)
        return (f"{kind}_make(({element})({real_code}), ({element})({imag_code}))", kind)


    def _emit_complex_compare(self, operator:str, left:str, lt:object, right:str,
                              rt:object) -> Optional[Translated]:
        """`z == w` / `z != w` as the header's equality test.

        numpy compares the two parts, so an equality is a real comparison of both. A real
        operand is widened (`c64 == np.float32(2)` is the comparison with `2 + 0i`), which is
        the same widening the arithmetic uses; anything else has no device meaning.
        """
        left_code, left_dtype = self._complex_operand(left, lt)
        right_code, right_dtype = self._complex_operand(right, rt)
        if left_dtype is None and right_dtype is None:
            return None
        left_side:str = self._complex_side(lt, left_dtype)
        right_side:str = self._complex_side(rt, right_dtype)
        widened:bool = "c128" in (left_side, right_side) or "double" in (left_side, right_side)
        kind:str = "c128" if widened else "c64"
        if left_dtype is not None and right_dtype is not None:
            if left_side == right_side:
                self._require_complex(left_dtype)
                comparison:str = f"{kind}_equal({left_code}, {right_code})"
            else:
                self.complex_needs.update(complex_support.mixed_needs())
                comparison = (f"{left_side}_equal_{right_side}"
                              f"({left_code}, {right_code})")
        else:
            complex_is_left:bool = left_dtype is not None
            scalar_side:str = right_side if complex_is_left else left_side
            complex_code:str = left_code if complex_is_left else right_code
            scalar_code:str = right_code if complex_is_left else left_code
            if scalar_side == "double" and not widened:
                self.complex_needs.update(complex_support.mixed_needs())
                comparison:str = f"{kind}_equal_c128({complex_code}, {scalar_code})"
            else:
                # The scalar is widened by building the complex value it means, which the
                # header spells as its own conversion.
                self._require_complex(np.dtype(np.complex128 if widened else np.complex64))
                widener:str = "c128_from_real" if widened else "c64_from_real"
                scalar_element:str = "double" if widened else "float"
                comparison = (f"{kind}_equal({complex_code},"
                              f" {widener}(({scalar_element})({scalar_code})))")
        if operator == "NotEq":
            return f"(!{comparison})", "bool"
        return f"({comparison})", "bool"


    def _emit_complex_method(self, attr:str, base_code:str, base_t:object,
                             dtype:np.dtype, node:ast.Call)->Optional[Translated]:
        """One of a complex value's own methods, as the header's function.

        numpy's complex scalars define `conjugate()` (and its alias `conj()`), plus
        `astype()` for a width change. Nothing else is a method -- the rest of the complex
        vocabulary is the module functions -- so a miss returns None and the caller names
        the methods that exist.
        """
        kind:str = self._complex_side(base_t, dtype)
        if not node.args and attr in ("conjugate", "conj"):
            self._require_complex(dtype)
            return f"{kind}_conjugate({base_code})", kind
        return None
