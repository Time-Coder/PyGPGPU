from __future__ import annotations
import ast
from typing import (Any, List, Tuple)
from .vectors import (
    _C_FIELD_TYPES,
    _vec_components,
    is_vec,
)
from .builtins import (
    PRINT_INT_CASTS,
    PRINT_INT_FORMATS,
    PRINT_MAX_LINES,
    PRINT_REAL_BIG,
    PRINT_REAL_DIGITS,
    PRINT_REAL_PLAIN_MAX,
    PRINT_REAL_SCI_HI,
    PRINT_REAL_SCI_LO,
    PRINT_REAL_SPECIAL_COUNT,
)
from .predicates import (
    is_int,
    is_ptr,
)


class _PrintEmitMixin:

    def _is_print_call(self, node:ast.expr)->bool:
        """True for a statement-level call of the builtin ``print``.

        Only the builtin counts: a module that defines its own print() means its own
        function, which is then translated (or rejected) like any other call. The builtin
        is not a module global, so it is looked up in the function's own namespace.
        """
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print" and self._print_is_builtin())


    def _print_is_builtin(self)->bool:
        """True while translating a function whose global scope does not shadow print."""
        namespace:object = getattr(self.func, "__globals__", None)
        if namespace is None:
            return True
        return namespace.get("print", print) is print


    def _emit_print(self, call:ast.Call)->None:
        """Translate ``print(...)`` into a device printf that prints the same text.

        Python's print writes to the host process's stdout, which a kernel has no access
        to; the device counterpart is printf. The rewrite mirrors python's text: arguments
        separated by one space, a trailing newline, and every value rendered the way
        ``str()`` renders it (True/False, decimal integers, python's shortest round-trip
        floats, and ``float4(1.0, 2.0, 3.0, 4.0)`` for vectors).

        One line becomes one printf call site per branch leaf, selected by a computed
        index. That keeps each line whole: a device printf call site has its own output
        buffer and the buffers are drained call site by call site, so a line assembled
        from several printf calls would come out split up and interleaved with other work
        items. Each leaf repeats the whole format string, so a line is never broken.

        The HIP backend is excluded on purpose. Its driver drops the device ("unspecified
        launch failure", followed by a Windows display-driver reset on the tested ROCm 7.2
        machine) once a kernel with this many printf call sites is launched, and debug
        output is not worth crashing the machine over. Use OpenCL or CUDA for printing.
        """
        # launch_backend, not backend: a hip kernel goes through cuda code generation, so
        # self.backend is "cuda" there and only the runtime tells the two apart.
        if self.launch_backend == "hip":
            self._error(
                "print() is not supported on the hip backend: launching a kernel with the "
                "generated printf call sites makes the driver drop the device on this "
                "runtime. Run the kernel with the opencl or cuda backend to see its output, "
                "or write results into an array and inspect them on the host", call)
        if self._metal_syntax:
            # The emitted call sites are OpenCL's: `printf` writes into a per-call-site
            # output buffer that the host drains after the launch. MSL has no such
            # buffer -- `printf` in Metal goes to a shared log that the command buffer
            # owns and no host code reads -- so the generated code would compile and
            # silently lose the output. Refuse rather than pretend, the same way the
            # HIP path above does.
            self._error(
                "print() is not supported on the metal backend: the generated call sites "
                "write into a per-call-site output buffer that only the OpenCL runtime "
                "drains, while Metal's printf goes to a log no host code here reads. Run "
                "the kernel with the opencl or cuda backend to see its output, or write "
                "results into an array and inspect them on the host", call)
        # sep/end are fixed text, so both can be folded into the format string. file and
        # flush are not translated: there is only one device output stream.
        separator, terminator = self._print_sep_end(call)

        pieces:List[Tuple[str, str, Any]] = []
        for index, argument in enumerate(call.args):
            if index:
                pieces.append(("text", separator, None))
            pieces.extend(self._print_argument(argument))

        # The digit-count code for a real argument has to run before the switch that picks
        # the leaf, so it is emitted as the pieces are expanded.
        leaves:List[Tuple[str, List[str]]] = [("", [])]
        selectors:List[Tuple[str, int]] = []
        self._expand_print_pieces(pieces, leaves, selectors)
        self._emit_print_leaves(leaves, selectors, call, terminator)


    def _print_fstring(self, node:ast.JoinedStr)->List[Tuple[str, str, Any]]:
        """An f-string as print pieces: its literal runs are text, its fields are values.

        `f"slot {i}"` is the same output as `print("slot", i, sep="")`, and that is how it is
        emitted -- python's f-string is a `JoinStr` of `Constant` and `FormattedValue` parts,
        and the second sort is exactly one value. A conversion (`!r`/`!s`/`!a`) or a format
        spec (`:>4`, `:.2f`) asks python's formatting machinery for text the device has no
        formatter for, so each is refused with the reason rather than silently ignored.
        """
        pieces:List[Tuple[str, str, Any]] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                pieces.append(("text", part.value, None))
                continue
            if not isinstance(part, ast.FormattedValue):
                self._error(
                    "an f-string part has no device form: use literal text and plain "
                    "`{expression}` fields", part)
            if part.conversion not in (-1, None):
                self._error(
                    "an f-string conversion (`!r`, `!s`, `!a`) formats through python's "
                    "repr/str, which the device has no formatter for: print the value in a "
                    "plain `{expression}` field instead", part)
            if part.format_spec is not None:
                self._error(
                    "an f-string format spec (`{value:>4}`, `{value:.2f}`) needs python's "
                    "formatting machinery, which the device does not have: print the value "
                    "in a plain `{expression}` field instead", part)
            pieces.extend(self._print_argument(part.value))
        return pieces


    def _print_sep_end(self, call:ast.Call)->Tuple[str, str]:
        """The separator and terminator a print() call asks for.

        Both become part of the format string, which has to be a compile-time constant on
        the device (OpenCL rejects a computed format), so they must be string literals.
        ``file`` and ``flush`` are refused: a kernel has one output stream and nothing to
        flush.
        """
        separator:str = " "
        terminator:str = "\n"
        for keyword in call.keywords:
            if keyword.arg is None:
                self._error(
                    "print() does not accept **kwargs: the printed text must be fixed in "
                    "the device code", call)
            if keyword.arg == "file":
                self._error(
                    "print(file=...) is not supported in a kernel: device printf writes to "
                    "the backend's own output stream", call)
            if keyword.arg == "flush":
                self._error(
                    "print(flush=...) has no meaning in a kernel: device output is "
                    "drained when the launch completes", call)
            if keyword.arg not in ("sep", "end"):
                self._error(
                    f"print() got an unexpected keyword argument '{keyword.arg}'; only sep "
                    f"and end are supported", call)
            if not (isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)):
                self._error(
                    f"print({keyword.arg}=...) must be a string literal: the format string "
                    f"is fixed in the device code, so it cannot be computed", call)
            if keyword.arg == "sep":
                separator = keyword.value.value
            else:
                terminator = keyword.value.value
        return separator, terminator


    def _print_argument(self, node:ast.expr)->List[Tuple[str, str, Any]]:
        """Split one print() argument into pieces.

        A piece is one of:
          ("text", literal, None)   fixed text, copied into the format string
          ("int", code, type)       an integer, printed by a printf conversion
          ("bool", code, "bool")    a bool, printed as True/False
          ("real", code, type)      a float or double, printed like python's repr
          ("vec", code, type)       a vector, printed like its python repr

        An f-string is exactly a list of those pieces already: its literal runs are fixed
        text and each `{expression}` is one value, so it is expanded here rather than
        refused. A conversion or a format spec (`{x!r}`, `{x:>4}`) is not, because python's
        formatting machinery has no device counterpart -- the value is printed its own way.
        """
        if isinstance(node, ast.JoinedStr):
            return self._print_fstring(node)
        if isinstance(node, ast.Constant) and node.value is None:
            return [("text", "None", None)]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # A literal string is fixed text, and the only way a string can reach print():
            # the translator has no string type.
            return [("text", node.value, None)]

        code, t = self._visit_expression(node)
        if is_vec(t):
            return [("vec", code, t)]
        if is_ptr(t):
            self._error(
                "print() cannot print an array: python prints the whole array's repr, which "
                "has no device counterpart. Print an element (print(a[i])) or a value instead",
                node)
        if t == "bool":
            return [("bool", code, "bool")]
        if t in ("float", "double"):
            return [("real", code, t)]
        if is_int(t):
            return [("int", code, t)]
        self._error(f"print() does not support a value of type {t}", node)


    def _expand_print_pieces(self, pieces:List[Tuple[str, str, Any]],
                             leaves:List[Tuple[str, List[str]]],
                             selectors:List[Tuple[str, int]])->None:
        """Fold pieces into leaves (one printf each), recording the selectors in order.

        Leaves are ordered so that the leaf index is the mixed-radix number formed by the
        selectors taken in order, which is what _emit_print_leaves relies on: each bool
        contributes a radix-2 digit, each real a digit of its own budget.
        """
        for kind, code, extra in pieces:
            if kind == "text":
                leaves[:] = [(fmt + self._c_string_text(code), args) for fmt, args in leaves]
            elif kind == "int":
                conversion:str = "%" + self._int_format(extra)
                argument:str = self._int_argument(code, extra)
                leaves[:] = [(fmt + conversion, args + [argument]) for fmt, args in leaves]
            elif kind == "bool":
                # Python prints a bool as True/False, and no printf conversion produces
                # that text from a computed value, so the line is split into the two
                # spellings. True is the first leaf, hence selector 0 for a true value.
                expanded:List[Tuple[str, List[str]]] = []
                for fmt, args in leaves:
                    expanded.append((fmt + "True", list(args)))
                    expanded.append((fmt + "False", list(args)))
                leaves[:] = expanded
                selectors.append((f"(({code}) ? 0 : 1)", 2))
            elif kind == "real":
                selector, budget = self._print_real_selector(code, extra)
                expanded = []
                for fmt, args in leaves:
                    for choice in range(budget):
                        text:str = self._print_real_text(choice)
                        conversion:str = self._print_real_conversion(choice, extra)
                        if text:
                            # A special value spells itself and takes no argument.
                            expanded.append((fmt + text, list(args)))
                        else:
                            expanded.append((fmt + conversion, list(args) + [f"((double)({code}))"]))
                leaves[:] = expanded
                selectors.append((selector, budget))
            elif kind == "component":
                n:str = self._next_temp()
                self._emit(f"double _pa{n} = fabs((double)({code}));")
                expanded = []
                for fmt, args in leaves:
                    for choice in (0, 1):
                        conversion:str = "%.1f" if choice == 0 else "%.17g"
                        expanded.append((fmt + conversion, list(args) + [f"((double)({code}))"]))
                leaves[:] = expanded
                selectors.append((f"((_pa{n} == floor(_pa{n}) && _pa{n} < 1e16) ? 0 : 1)", 2))
            else:
                sub_pieces:List[Tuple[str, str, Any]] = self._print_vec_pieces(code, extra)
                sub_leaves:List[Tuple[str, List[str]]] = [("", [])]
                sub_selectors:List[Tuple[str, int]] = []
                self._expand_print_pieces(sub_pieces, sub_leaves, sub_selectors)
                expanded = []
                for fmt, args in leaves:
                    for sub_fmt, sub_args in sub_leaves:
                        expanded.append((fmt + sub_fmt, args + sub_args))
                leaves[:] = expanded
                selectors.extend(sub_selectors)


    def _print_vec_pieces(self, code:str, vec:str)->List[Tuple[str, str, Any]]:
        """The pieces of a vector's python repr: "float4(" + components + ")".

        A vector's repr renders each component as a Python float, i.e. the exact double the
        component widens to. Every component therefore gets the same two shapes: a whole
        value keeps the ".0" python shows, and everything else is printed with 17
        significant digits, which is what the widened float32 needs to read back exactly
        (the low bits of such a double are zero, so 17 digits are its shortest form too).
        Keeping the layout that small is what lets a whole vector stay one printf.
        """
        pieces:List[Tuple[str, str, Any]] = [("text", f"{vec}(", None)]
        for position, (field_name, _field_type) in enumerate(_vec_components(vec)):
            if position:
                pieces.append(("text", ", ", None))
            component_t:str = _C_FIELD_TYPES[(vec, field_name)]
            component:str = f"({code}).{field_name}"
            if component_t in ("float", "double"):
                pieces.append(("component", component, "double"))
            elif component_t == "bool":
                pieces.append(("bool", component, "bool"))
            else:
                pieces.append(("int", component, component_t))
        pieces.append(("text", ")", None))
        return pieces


    def _print_real_conversion(self, choice:int, t:str)->str:
        """The printf conversion for one leaf of a real argument, or its fixed text.

        The leaves are a fixed layout, so a single integer selects one:
          0..4        nan, inf, -inf, 0.0 and -0.0: python's exact spelling, so the leaf
                      carries the text and no argument
          5           a whole number below the notation threshold: python prints 2.0
                      where %g would print 2
          6..6+max-1  the "%g" significant-digit budgets, 1..max: inside python's
                      fixed-notation band %g switches notation exactly as python does
          ..          the "%.Ne" budgets, 0..max-1 decimals, for the band where python
                      writes the exponent form but a "%g" of that budget would not
        """
        if choice < PRINT_REAL_SPECIAL_COUNT:
            return ""
        if choice == PRINT_REAL_SPECIAL_COUNT:
            return "%.1f"
        max_digits:int = PRINT_REAL_DIGITS[t == "float"]
        offset:int = choice - PRINT_REAL_SPECIAL_COUNT - 1
        if offset < max_digits:
            return "%." + str(offset + 1) + "g"
        # An exponent leaf prints the digits "%g" of that budget would print, only with the
        # exponent form made mandatory: p significant digits are p-1 decimals. A value at
        # the very top of the precision's range needs no decimals, so this starts at 0.
        decimals:int = offset - max_digits
        return "%." + str(max(decimals, 0)) + "e"


    def _print_real_text(self, choice:int)->str:
        """The fixed text of a special leaf (nan, inf, -inf, 0.0, -0.0); otherwise empty."""
        specials:List[str] = ["nan", "inf", "-inf", "0.0", "-0.0"]
        return specials[choice] if 0 <= choice < len(specials) else ""


    def _print_real_selector(self, code:str, t:str)->Tuple[str, int]:
        """Emit the leaf selection for a real argument and return (selector, leaf count).

        Python prints a real with the fewest significant digits that read back as the same
        value, so the digit count depends on the value: it is found here, at run time, by
        taking the first budget whose rounded value lands within half the gap to the
        neighbouring value the argument's own precision can hold.

        The gap is taken from the argument's binary exponent (ldexp, because OpenCL C has no
        nextafter), and the rounding is done on the value itself, scaled by 10^(budget-1-e).
        Both powers are built from two factors so the intermediate stays in range even for
        the smallest and largest values.
        """
        is_float_value:bool = (t == "float")
        max_digits:int = PRINT_REAL_DIGITS[is_float_value]
        plain_max:str = PRINT_REAL_PLAIN_MAX[is_float_value]
        sci_hi:int = PRINT_REAL_SCI_HI[is_float_value]
        big:str = PRINT_REAL_BIG[is_float_value]
        whole_leaf:int = PRINT_REAL_SPECIAL_COUNT
        general_base:int = whole_leaf + 1
        exponent_base:int = general_base + max_digits
        # The gap to the next representable value: one float mantissa step, or one double.
        gap:str = ("ldexp(1.0, ilogb((float)_pa%s) - 23)" if is_float_value
                   else "ldexp(1.0, ilogb(_pa%s) - 52)")
        smallest_normal:str = ("1.17549435e-38" if is_float_value else "2.2250738585072014e-308")
        smallest_step:str = ("1.40129846e-45" if is_float_value else "4.9406564584124654e-324")
        n:str = self._next_temp()
        # These temporaries stay in the current block: the switch that follows the
        # expansion is emitted at the same level and reads the selector out of them.
        self._emit(f"double _pv{n} = (double)({code});")
        self._emit(f"double _pa{n} = fabs(_pv{n});")
        self._emit(f"int _pd{n} = {general_base};")
        # The five values python spells out own the first leaves.
        self._emit(f"if (!(_pa{n} <= {big}))"
                   f" _pd{n} = (_pa{n} != _pa{n}) ? 0 : (_pv{n} < 0.0 ? 2 : 1);")
        self._emit(f"else if (_pa{n} == 0.0) _pd{n} = signbit(_pv{n}) ? 4 : 3;")
        self._emit(f"else if (_pa{n} == floor(_pa{n}) && _pa{n} < {plain_max})"
                   f" _pd{n} = {whole_leaf};")
        self._emit("else")
        self._emit("{")
        self._indent += 1
        self._emit(f"double _pgap{n} = {gap % n};")
        # A power of two has only half that gap below it.
        self._emit(f"if (_pa{n} == ldexp(1.0, ilogb(_pa{n}))) _pgap{n} *= 0.5;")
        # A subnormal value has a fixed smallest step, so ldexp would underflow it.
        self._emit(f"if (_pa{n} < {smallest_normal}) _pgap{n} = {smallest_step};")
        self._emit(f"int _pe{n} = (int)floor(log10(_pa{n}));")
        self._emit(f"double _ps{n} = pow(10.0, (double)_pe{n});")
        self._emit(f"if (_ps{n} > _pa{n})")
        self._emit("{")
        self._emit(f"    _pe{n} -= 1; _ps{n} = pow(10.0, (double)_pe{n});")
        self._emit("}")
        self._emit(f"else if (_ps{n} * 10.0 <= _pa{n})")
        self._emit("{")
        self._emit(f"    _pe{n} += 1; _ps{n} = pow(10.0, (double)_pe{n});")
        self._emit("}")
        # Half the gap either side, then the shortest budget whose rounding fits inside it.
        self._emit(f"double _plo{n} = _pa{n} - 0.5 * _pgap{n};")
        self._emit(f"double _phi{n} = _pa{n} + 0.5 * _pgap{n};")
        self._emit(f"int _pbest{n} = {max_digits};")
        self._emit(f"for (int _pk{n} = 0; _pk{n} < {max_digits}; _pk{n}++)")
        self._emit("{")
        self._indent += 1
        # Round to _pk+1 significant digits: divide by 10^(e-_pk), round, multiply back.
        # The exponent is split in two so no factor leaves the double range.
        self._emit(f"int _pstep{n} = _pe{n} - _pk{n};")
        self._emit(f"int _phalf{n} = _pstep{n} / 2;")
        self._emit(f"double _pdown{n} = pow(10.0, (double)_phalf{n})"
                   f" * pow(10.0, (double)(_pstep{n} - _phalf{n}));")
        self._emit(f"double _pq{n} = round(_pa{n} / _pdown{n}) * _pdown{n};")
        self._emit(f"if (_pq{n} >= _plo{n} && _pq{n} <= _phi{n})")
        self._emit("{")
        self._emit(f"    _pbest{n} = _pk{n} + 1; break;")
        self._emit("}")
        self._indent -= 1
        self._emit("}")
        self._emit_blank()
        # Python's notation bands: outside them the text is in the exponent form, inside
        # them "%g" keeps the fixed notation python uses. The exponent leaf with budget p
        # prints exactly the digits of "%.pg", so the same budget selects both.
        sci_lo:int = PRINT_REAL_SCI_LO[is_float_value]
        self._emit(f"if (_pe{n} <= {sci_lo} || _pe{n} >= {sci_hi})"
                   f" _pd{n} = {exponent_base} + _pbest{n} - 1;")
        self._emit(f"else _pd{n} = {general_base} + _pbest{n} - 1;")
        self._indent -= 1
        self._emit("}")
        # The last leaf is index exponent_base + max_digits - 1, so the line needs one more.
        return f"_pd{n}", exponent_base + max_digits


    def _emit_print_leaves(self, leaves:List[Tuple[str, List[str]]],
                           selectors:List[Tuple[str, int]], node:ast.Call,
                           terminator:str="\n")->None:
        """Emit the leaves as printfs selected by the mixed-radix leaf index."""
        if len(leaves) > PRINT_MAX_LINES:
            self._error(
                f"print() would need {len(leaves)} printf calls for this one line, above the "
                f"{PRINT_MAX_LINES} a kernel allows: too many booleans or floats in one "
                f"line. Split the text over several print() calls", node)

        tail:str = self._c_string_text(terminator)
        if len(leaves) == 1 and not selectors:
            format_text:str = leaves[0][0] + tail
            self._emit('printf("%s"%s);' % (format_text, (", " + ", ".join(leaves[0][1])) if leaves[0][1] else ""))
            return

        # Mixed-radix index over the selectors, in the order the leaves were expanded.
        index:str = "0"
        for selector, radix in selectors:
            index = f"(({index}) * {radix} + ({selector}))"
        cases:List[str] = []
        for position, (format_text, args) in enumerate(leaves):
            arguments:str = (", " + ", ".join(args)) if args else ""
            cases.append('case %d: printf("%s%s"%s); break;'
                         % (position, format_text, tail, arguments))
        cases.append("default: break;")
        self._emit(f"switch ((int){index})")
        self._emit("{")
        self._emit("\n".join(f"    {case}" for case in cases))
        self._emit("}")


    def _int_format(self, t:str)->str:
        """The printf conversion for an integer type on this backend."""
        return PRINT_INT_FORMATS[self.printf_backend].get(t, "d")


    def _int_argument(self, code:str, t:str)->str:
        """The C expression the printf conversion for an integer type expects."""
        cast:str = PRINT_INT_CASTS[self.printf_backend].get(t, "int")
        return f"({cast})({code})"


    def _c_string_text(self, text:str)->str:
        """Escape text for a C string literal, keeping % a literal percent."""
        escaped:str = text.replace("\\", "\\\\").replace('"', '\\"')
        escaped = escaped.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        return escaped.replace("%", "%%")
