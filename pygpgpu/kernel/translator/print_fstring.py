# An f-string's numeric format spec, as the printf conversion that prints the same text.
from __future__ import annotations
import ast
import re
from typing import (Any, List, Tuple)
from .predicates import (
    is_int,
)


# python's format spec, in the subset printf can spell:
#     [[fill]align][sign][0][width][.precision][type]
# `align` is restricted to `<`/`>` because printf's `-` flag is the only alignment it has
# (`^` centring has no conversion), and `fill` is not captured: a non-space fill would need
# padding the device cannot emit, which the caller gets told about by the refusal.
_FORMAT_SPEC = re.compile(
    r"^(?:(?P<align>[<>]))?"
    r"(?P<sign>[+\- ])?"
    r"(?P<zero>0)?"
    r"(?P<width>\d+)?"
    r"(?:\.(?P<precision>\d+))?"
    r"(?P<type>[fFeEgGd])?$")


class _PrintFStringMixin:
    """The `print(f"...")` half of the print emitter: its fields and their format specs."""

    def _print_fstring(self, node:ast.JoinedStr)->List[Tuple[str, str, Any]]:
        """An f-string as print pieces: its literal runs are text, its fields are values.

        `f"slot {i}"` is the same output as `print("slot", i, sep="")`, and that is how it is
        emitted -- python's f-string is a `JoinStr` of `Constant` and `FormattedValue` parts,
        and the second sort is exactly one value.

        A numeric format spec (`{value:.2f}`, `{i:4d}`, `{i:<6d}`) is folded into the piece
        as a printf conversion, because printf has the spelling the spec asks for and the
        device formats the value inline. A spec printf cannot express (a `,` thousands
        separator, a `%` scale, a string field) is refused by name -- silently printing the
        value another way would be worse than refusing it. A conversion (`!r`/`!s`/`!a`) is
        still refused: it goes through python's repr/str, which the device does not have.
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
            spec:str = ""
            if part.format_spec is not None:
                spec = self._fstring_format_spec(part)
            pieces.extend(self._print_argument(part.value, spec))
        return pieces


    def _fstring_format_spec(self, part:ast.FormattedValue)->str:
        """The literal text of a field's format spec, or a refusal naming what cannot map.

        A spec is a `JoinedStr` of constants in practice (a spec computed from a value would
        be a computed format string, which printf cannot take), so only literal text is
        accepted. What reaches printf is checked in `_printf_conversion`, which is where the
        supported shape is decided.
        """
        spec_node:ast.expr = part.format_spec  # type: ignore[assignment]
        if isinstance(spec_node, ast.JoinedStr):
            texts:List[str] = []
            for piece in spec_node.values:
                if not (isinstance(piece, ast.Constant) and isinstance(piece.value, str)):
                    self._error(
                        "an f-string format spec must be literal text: the device printf "
                        "format string is fixed in the generated code, so it cannot be "
                        "computed from a value", piece)
                texts.append(piece.value)
            return "".join(texts)
        if isinstance(spec_node, ast.Constant) and isinstance(spec_node.value, str):
            return spec_node.value
        self._error(
            "an f-string format spec must be literal text: the device printf format string "
            "is fixed in the generated code, so it cannot be computed from a value", spec_node)
        return ""


    def _printf_conversion(self, spec:str, value_type:object, node:ast.AST)->str:
        """A python format spec as the printf conversion that prints the same text.

        python's mini-language and printf agree on the part a debug print uses: an optional
        fill/alignment, an optional width, an optional precision, and a type letter (`f`,
        `e`, `g`, `d`). The supported shape is:

            [[fill]align][sign][0][width][.precision][type]

        with `align` in `<`/`>` and `type` in `f`/`e`/`g`/`d`, which maps one-for-one onto
        printf's flags. Two specs python has and printf does not are refused rather than
        approximated: a `,` thousands separator and a `%` scale. A field with no type letter
        takes the one its value's type implies, which is what python does.
        """
        if "," in spec:
            self._error(
                "an f-string format spec with a `,` thousands separator has no device "
                "spelling: printf has no grouped-digits conversion, so printing it here "
                "would print an ungrouped number. Format it on the host instead", node)
        if spec.endswith("%"):
            self._error(
                "an f-string format spec ending in `%` scales the value by 100, which printf "
                "cannot do: multiply by 100.0 in the body and print that with `:.2f` instead",
                node)
        match = _FORMAT_SPEC.match(spec)
        if match is None:
            self._error(
                f"the f-string format spec ':{spec}' has no device spelling: the supported "
                f"shape is [[fill]align][sign][0][width][.precision][type] with type one of "
                f"f/e/g/d (a `,` separator, a `%` scale and a string field are refused)",
                node)
        align:str = match.group("align") or ""
        sign:str = match.group("sign") or ""
        zero:str = match.group("zero") or ""
        width:str = match.group("width") or ""
        precision:str = match.group("precision")
        type_letter:str = match.group("type") or ""

        if align == "<":
            flag:str = "-"
        elif align == ">":
            flag = ""
        else:
            flag = ""
        if zero and not flag:
            flag = "0"
        if sign == "+":
            flag += "+"
        elif sign == " ":
            flag += " "

        if type_letter == "":
            # python's default: `d` for an integer, and the `g`-shaped repr for a real --
            # but an explicit spec means the caller wants fixed digits, so `f`.
            type_letter = "d" if is_int(value_type) else "f"
        if type_letter == "d" and not is_int(value_type):
            self._error(
                f"the f-string format spec ':{spec}' asks for an integer field, but this "
                f"value is a {value_type}; print it with `f`/`e`/`g` instead", node)
        if type_letter in ("f", "e", "g") and is_int(value_type):
            self._error(
                f"the f-string format spec ':{spec}' asks for a real field, but this value "
                f"is a {value_type}; print it with `d` instead", node)
        digits:str = "" if precision is None else f".{precision}"
        return f"%{flag}{width}{digits}{type_letter}"
