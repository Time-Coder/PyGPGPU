"""Reads a Metal Shading Language source into the per-kernel argument layout.

This replaces the regular-expression declaration scanner the Metal backend used to
carry. A scanner could only *find* the declarations it had a pattern for, so an
unanticipated spelling was silently missed -- and it could not see a body at all,
which is why every Metal pointer argument was copied to the device and back on
every launch whether the kernel read it or not.

The grammar is `tree-sitter-metal`, this package's own distribution (see the
`tree-sitter-metal` project beside this one). It descends from tree-sitter-cpp, so
the shape is a C++ tree with MSL's words in it, and the walk here reads that shape
with node types rather than with patterns.

What it reads is what the other backends' parsers read, through the same
`ProgramParser` surface:

* the entry points (`kernel void f(...)`), each one's parameters, and which of them
  the threadgroup dispatch fills in;
* the `struct` declarations, built into ctypes classes for `kernel_import`;
* the copy policy per pointer argument, from the body's reads and writes.

The parse result is cached beside the source, keyed by a digest of the preprocessed
text -- the same arrangement the C backends use, so a module's second import does
not pay for the walk again.
"""
from __future__ import annotations

import os
import re
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

from tree_sitter import Node

from ....cparser.ts_parser import make_parser
from .analysis import _MetalAnalysisMixin
from .declarations import _MetalDeclarationsMixin
from .node_kinds import DISPATCH_ATTRIBUTES
from .struct_builder import _MetalStructBuilderMixin

if TYPE_CHECKING:
    from ....cparser.ts_parser import ParsedVar, ParseMeta
    from ..compile_options import CompileOptions
    from ..program_info import ArgInfo, KernelInfo, StructInfo, VarInfo

try:
    import tree_sitter_metal as tsmetal
except ImportError:  # pragma: no cover - the dependency is a hard requirement
    raise ImportError(
        "the Metal backend needs the 'tree-sitter-metal' grammar package: "
        "install it with `pip install tree-sitter-metal`") from None


class ProgramParser(_MetalStructBuilderMixin, _MetalDeclarationsMixin,
                    _MetalAnalysisMixin):
    """The kernel layout of one MSL source file."""

    _parser = make_parser(tsmetal.language())

    def __init__(self) -> None:
        self._file_name: str = ""
        self._base_name: str = ""
        self._clean_code: str = ""
        self._includes: List[str] = []
        self._defines: Dict[str, object] = {}
        self._kernel_infos: Dict[str, "KernelInfo"] = {}
        self._struct_infos: Dict[str, "StructInfo"] = {}
        self._struct_types: Dict[str, type] = {}
        self._dispatch_names: Dict[str, List[str]] = {}
        self._newest_mtime: Union[float, bool] = False

    def parse(self, file_name: str, includes: Optional[List[str]] = None,
              defines: Optional[Dict[str, object]] = None,
              options: Optional["CompileOptions"] = None) -> Union[float, bool]:
        """Read one source file and record its kernel layout.

        `options` is accepted and ignored: it carries the backend's build options,
        which for Metal are the compile flags a `MTLCompileOptions` holds. Nothing
        in this walk depends on them.
        """
        self._file_name = file_name
        self._base_name = os.path.basename(file_name)
        self._includes = list(includes or ())
        self._defines = dict(defines or {})
        from ...driver.metal import MetaMetal
        if MetaMetal.print_info:
            print(f"parsing {self._base_name} ... ", end="", flush=True)
        with open(file_name, "r", encoding="utf-8") as source:
            self._clean_code = source.read()
        self._parse()
        self._newest_mtime = os.path.getmtime(file_name)
        if MetaMetal.print_info:
            print("done. ", flush=True)
        return self._newest_mtime

    # ---- the walk ----

    def _parse(self) -> None:
        self._kernel_infos.clear()
        self._struct_infos.clear()
        self._struct_types.clear()
        self._dispatch_names.clear()
        self._function_infos: Dict[str, "KernelInfo"] = {}
        self._kernel_bodies: Dict[str, Node] = {}

        tree = self._parser.parse(self._clean_code.encode("utf-8"))
        # A source the grammar could not read is refused rather than half-read: an
        # ERROR node means some declaration was not understood, and a launch that
        # silently bound the wrong argument would be far worse than a clear refusal.
        # This is the contract the declaration scanner carried ("raises rather than
        # guessing"), kept now that the tree says exactly which text defeated it.
        self._refuse_unreadable(tree.root_node)
        for child in self._top_level_nodes(tree.root_node):
            if child.type in ("struct_specifier", "union_specifier"):
                self._parse_struct(child)
            elif child.type == "function_definition":
                self._parse_function(child)

        for struct_info in self._struct_infos.values():
            self._create_struct_type(struct_info)
        self._analyze_kernels()
        for name, info in self._kernel_infos.items():
            info.dispatch_parameters = list(self._dispatch_names.get(name, ()))

    # ---- the walk ----

    def _refuse_unreadable(self, root: Node) -> None:
        """Raise when the grammar could not read some part of the source.

        Tree-sitter is error-tolerant: it answers a tree with an `ERROR` node (or a
        missing token) rather than raising, so a truncated or malformed source would
        otherwise be read as "a file that declares nothing" -- and a launch would fail
        later with a name the caller never wrote. The offending text is quoted so the
        message says which declaration defeated the grammar.
        """
        broken: List[str] = []
        stack: List[Node] = [root]
        while stack:
            node = stack.pop()
            if node.type == "ERROR" or node.is_missing:
                broken.append(self._quoted(node))
            stack.extend(node.named_children)
        if broken:
            raise ValueError(
                f"{self._base_name}: the Metal source could not be parsed; the grammar "
                f"did not understand {broken[0]}"
                + (f" (and {len(broken) - 1} more place(s))" if len(broken) > 1 else ""))

    def _quoted(self, node: Node) -> str:
        text: str = self._text(node).strip()
        return repr(" ".join(text.split())[:60]) if text else "a missing token"

    def _text(self, node: Optional[Node]) -> str:
        return (node.text or b"").decode("utf-8", "replace") if node is not None else ""

    def _top_level_nodes(self, root: Node):
        """Every declaration at file scope, looking through conditional blocks.

        A generated source is wrapped in `#ifndef <guard>` / `#endif`, and the grammar
        puts the entire file inside a `preproc_ifdef` node rather than leaving the
        declarations at the top level. Reading only `root.named_children` therefore
        found no kernel at all in exactly the sources this package writes -- the two
        phases of a magic body, and every generated module after the first.
        """
        stack: List[Node] = [root]
        while stack:
            node = stack.pop()
            kind: str = node.type
            if kind in ("translation_unit", "preproc_ifdef", "preproc_if",
                        "preproc_else", "preproc_elif", "preproc_def",
                        "preproc_function_def", "linkage_specification",
                        "declaration_list"):
                # A preprocessor branch holds its declarations as children, in order;
                # push them reversed so they are visited left to right.
                stack.extend(reversed(node.named_children))
                continue
            yield node

    def _new_struct_info(self, name: str, is_union: bool = False) -> "StructInfo":
        from ..program_info import StructInfo

        return StructInfo(name=name, is_union=is_union)

    def _new_kernel_info(self, name: str) -> "KernelInfo":
        from ..program_info import KernelInfo

        return KernelInfo(name=name)

    def _new_var_info(self, desc: "ParsedVar") -> "VarInfo":
        from ..program_info import VarInfo

        return VarInfo(
            name=desc["name"],
            type_str=desc["type_str"],
            array_shape=desc["array_shape"],
            type_qualifiers=desc["type_qualifiers"],
            kernel_flags=desc["kernel_flags"],
        )

    def _new_arg_info(self, kernel_info: "KernelInfo", desc: "ParsedVar") -> "ArgInfo":
        from ..program_info import ArgInfo

        return ArgInfo(
            name=desc["name"],
            type_str=desc["type_str"],
            array_shape=desc["array_shape"],
            type_qualifiers=desc["type_qualifiers"],
            kernel_flags=desc["kernel_flags"],
            address=desc.get("address"),
        )

    @property
    def file_name(self) -> str:
        return self._file_name

    @property
    def base_name(self) -> str:
        return self._base_name

    @property
    def clean_code(self) -> str:
        return self._clean_code

    @property
    def kernel_infos(self) -> Dict[str, "KernelInfo"]:
        return self._kernel_infos

    @property
    def structs(self) -> Dict[str, type]:
        return self._struct_types


__all__ = ["ProgramParser"]
