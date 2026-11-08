from __future__ import annotations
import copy
import re
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from tree_sitter import Language, Node, Parser
from ...constants import KernelFlags
from .ts_declarations import _TsDeclarationsMixin
from .ts_analysis import _TsAnalysisMixin
from .ts_body import _TsBodyMixin
from .ts_expressions import _TsExpressionsMixin


if TYPE_CHECKING:
    from typing_extensions import Protocol, TypedDict

    class ParsedVar(TypedDict, total=False):
        """A variable or parameter as the tree walk describes it.

        Built in two stages: `_parse_declaration_head` fills the type-level fields every
        declaration shares, and `_resolve_declarator` adds the ones that depend on the
        individual declarator. Every field is present by the time a parser reads the dict,
        so `total=False` describes the intermediate state rather than a half-usable value.
        `address`, `access` and `kernel_flags` are backend-neutral by design -- each parser
        maps them onto its own enum.
        """
        name: str
        type_str: str
        base_type: str
        array_shape: Tuple[int, ...]
        is_ptr: bool
        address: Optional[str]
        access: Optional[str]
        type_qualifiers: List[str]
        kernel_flags: KernelFlags

    class ParseMeta(TypedDict):
        """What a parser caches beside a compiled source, and reads back on the next run.

        `_save`/`load_var` round-trip it, so the five keys are the whole record: the
        preprocessed text, the files it pulled in, the line remapping for diagnostics,
        and the two tables the launch path needs. The value types are the per-backend
        records, named here by the shapes the walk relies on.
        """
        clean_code: str
        related_files: Set[str]
        line_map: Dict[int, str]
        struct_infos: Dict[str, "StructInfoLike"]
        kernel_infos: Dict[str, "KernelInfoLike"]

    class StructInfoLike(Protocol):
        """What the tree walk needs from a backend's structure record."""
        name: str
        is_union: bool
        members: Dict[str, "VarInfoLike"]

    class VarInfoLike(Protocol):
        """What the analysis needs from a backend's variable record."""
        name: str
        type_str: str
        array_shape: Tuple[int, ...]
        kernel_flags: KernelFlags

    class ArgInfoLike(VarInfoLike, Protocol):
        """A parameter record: a variable that also belongs to a kernel."""
        member_infos: Dict[str, VarInfoLike]

    class KernelInfoLike(Protocol):
        """What the tree walk needs from a backend's kernel record."""
        name: str
        args: Dict[str, ArgInfoLike]

    class AnalysisContext(TypedDict):
        """The read/write analysis's running state, one dict for the whole body walk."""
        reads: Set[str]
        writes: Set[str]
        escapes: Set[str]
        member_reads: Dict[str, Set[str]]
        member_writes: Dict[str, Set[str]]
        struct_args: Set[str]


def make_parser(language: Language) -> Parser:
    """Create a tree-sitter Parser configured for ``language``.

    ``language`` may be either an already-built ``tree_sitter.Language`` (the
    form returned by the OpenCL/CUDA grammar distributions used here) or a raw
    ``PyCapsule`` such as the one the classic ``tree-sitter-cuda`` binding
    returns; it is coerced into a ``tree_sitter.Language`` on the modern line.

    The tree-sitter Python bindings changed across releases: the legacy
    0.20.x line (the last line shipping CPython 3.7+ wheels) requires
    ``Parser(); parser.set_language(language)`` and its ``Parser(language)``
    constructor silently produces a parser that fails at parse time, whereas
    modern 0.21+ (which only installs on CPython 3.9+) has ``set_language``
    removed and requires ``Parser(language)``. This helper picks the correct
    construction for whichever tree-sitter is importable, so consumer code can
    be written once and run on Python 3.7 or newer.
    """
    if not isinstance(language, Language):
        # A capsule is what the classic tree-sitter-cuda binding exposes on the
        # modern line; wrap it so both old and new callers get a Language.
        language = Language(language)
    if hasattr(Parser, "set_language"):
        parser = Parser()
        parser.set_language(language)
        return parser
    return Parser(language)








class TSParserMixin(_TsDeclarationsMixin, _TsAnalysisMixin, _TsBodyMixin, _TsExpressionsMixin):

    @staticmethod
    def _node_text(node:Node)->str:
        """Decode a tree-sitter node's source text.

        `Node.text` is Optional in the bindings (it is None for a node that no longer
        references its original tree), so callers must not decode it directly.
        """
        return (node.text or b"").decode()


    def _ts_parse(self)->None:
        self._function_infos:Dict[str, KernelInfoLike] = {}
        self._kernel_bodies:Dict[str, Node] = {}
        self._helper_flags:Dict[str, Optional[Dict[str, KernelFlags]]] = {}
        self._analyzing_funcs:Set[str] = set()
        self._struct_aliases:Dict[str, str] = {}

        parse_code:str = (self._parse_code if getattr(self, "_parse_code", None) else self._clean_code)
        ast = self._parser.parse(parse_code.encode())
        for child in ast.root_node.named_children:
            if child.type == "type_definition":
                specifier, alias = child.named_children[0], (self._node_text(child.named_children[1]) if len(child.named_children) > 1 else "")
                if specifier.type in ("struct_specifier", "union_specifier"):
                    self._parse_struct(specifier, alias)
            elif child.type in ("struct_specifier", "union_specifier"):
                self._parse_struct(child)
            elif child.type == "function_definition":
                self._parse_function(child)

        for alias_name, target_name in self._struct_aliases.items():
            target_name = self._struct_aliases.get(target_name, target_name)
            if target_name in self._struct_infos:
                self._struct_infos[alias_name] = self._struct_infos[target_name]


__all__ = [
    "TSParserMixin",
    "_alias_source",
    "_analyze_body",
    "_analyze_kernel",
    "_analyze_kernels",
    "_apply_member_flags",
    "_bind_symbol",
    "_call_flags",
    "_collect_addr_taken_roots",
    "_collect_root_refs",
    "_flags_from_desc",
    "_function_body",
    "_handle_declaration",
    "_node_text",
    "_param_flags",
    "_parse_declaration_head",
    "_parse_function",
    "_parse_struct",
    "_pointer_members",
    "_resolve",
    "_resolve_declarator",
    "_resolve_expr_root",
    "_specifier_text",
    "_ts_parse",
    "_visit_call",
    "_visit_expression",
    "_visit_statement",
    "_visit_type_descriptor",
    "_walk_declarator",
    "make_parser",
]
