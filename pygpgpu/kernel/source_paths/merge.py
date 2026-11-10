# Merging rendered definitions into a file, and splitting one back into its pieces.
from __future__ import annotations
from .guards import (
    PROTOTYPE_MARKER,
    QUAT_DEFINE_PREFIX,
    _is_file_guard_line,
    _is_fixed_header_include,
    _is_include_only,
    _strip_file_guard,
    guard_macro,
)
from typing import Dict, List, Optional, Tuple



# The first word of a line that *continues* a body rather than starting a definition. A
# header line is a declaration (`c64 helper(c64 value)`), while the middle of a generated body
# reads `if (((b < 768)))` or `if (atomic_cmpxchg(...) == 0)` -- both contain a `(` and no `;`,
# which is otherwise exactly a header line's shape.
_CONTROL_KEYWORDS: set = {"if", "else", "for", "while", "do", "switch", "case",
                          "return", "break", "continue"}



def _definition_identity(code: str) -> Optional[str]:
    """A stable identity for one chunk of generated device code.

    A chunk is either a whole C function (`<qualifiers> <type> <name>(<params>)` followed by
    its body) or a structure typedef block. The function's own header line identifies it
    (the same helper specialized twice differs in the parameters, so those stay distinct);
    a structure is identified by the type name it declares, because its member list is part
    of the same declaration and must not be emitted twice.

    **Only a declaration line answers.** A line inside a body can have the header line's shape
    -- `if (atomic_cmpxchg(...) == 0)` has a `(` and no `;`, exactly like `c64 helper(c64 v)`
    -- and the file is split on blank lines, so a generated body containing one is several
    pieces. Reading such a piece as a definition of its own would give a body's middle an
    identity, which `_merge_definitions` groups by: the definition would then be *two*
    definitions, and refreshing it appended the second half instead of replacing the first --
    a file the compiler refuses with "extraneous closing brace". A declaration is therefore a
    line whose first word is not a control statement and which has no `=` before its opening
    parenthesis.

    The forward-declaration block answers the marker: it is one identity for the whole
    block, so `_merge_definitions` keeps it where it was and compares it as a whole --
    which is what lets two calls union their declarations in place instead of stacking a
    second block beside the first (see `merge_prototype_blocks`).
    """
    if code.lstrip().startswith(PROTOTYPE_MARKER):
        return PROTOTYPE_MARKER
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("struct "):
            return stripped
        if stripped.startswith("typedef struct"):
            # OpenCL spells a structure as `typedef struct { ... } Name;`, on one line or on
            # several. The one-line spelling ends here, and the whole declaration is its
            # identity -- it names the type, and the member list is part of that same
            # declaration. The multi-line one is caught by the reversed scan below, whose
            # `} Name;` is the type name it closes.
            if stripped.endswith(";"):
                return stripped
            continue
        if "(" in stripped and ";" not in stripped:
            head: str = stripped.split("(", 1)[0].strip()
            words: List[str] = head.split()
            if not words or words[0] in _CONTROL_KEYWORDS or "=" in head:
                continue
            return stripped
    # An OpenCL structure block ends in its own name; that is its identity.
    for line in reversed(code.splitlines()):
        stripped = line.strip()
        if stripped.startswith("}") and stripped.endswith(";"):
            return stripped
    return None



def _merge_definitions(existing: List[str], definitions: List[str]) -> Tuple[List[str], bool]:
    """`existing` with `definitions` applied: a changed definition replaces the one held.

    A generated file holds several kernels of one module, and each is written on its own
    call, so a definition the file already holds is normally skipped -- that is what makes a
    recompiled kernel idempotent. But "already holds" has to mean **the same text**, not the
    same signature: the body changes while the header line does not whenever *this package*
    starts generating something else for the same python source. A python value's width
    following a different mapping, a builtin emitting a different call, a fix to the
    emitter -- all of them keep the definition's name (the name hashes the signature, not
    the body) and change what it says. Skipping there left the file describing code the
    package no longer generates, and the file is what the compiler reads: the old body kept
    running. The python-source digest cannot catch it either, because the caller's `.py` did
    not move.

    Two units have to be kept apart. `definitions` is what the caller rendered: one element
    is one definition, whatever blank lines its body holds. `existing` is a *file*, split on
    blank lines, so the rest of a body that holds one is a piece of its own with no identity
    (the generated bounds guard emits exactly that shape). The pieces are therefore grouped
    behind the definition they belong to before anything is compared, and a changed
    definition replaces its whole group **in place** -- the order the kernels were written in
    is part of the file's shape (callee before caller), so a replacement must not move a
    definition to the end. A definition whose identity is unknown is appended, as before.

    Returns the merged pieces and whether anything changed, so a caller can skip a no-op
    write.
    """
    groups: List[Tuple[Optional[str], List[str]]] = []
    for piece in existing:
        identity:Optional[str] = _definition_identity(piece)
        if identity is None and groups and groups[-1][0] is not None:
            groups[-1][1].append(piece)
            continue
        groups.append((identity, [piece]))
    places: Dict[str, int] = {}
    for index, (identity, _pieces) in enumerate(groups):
        if identity is not None and identity not in places:
            places[identity] = index
    changed: bool = False
    added: List[str] = []
    added_identities: set = set()
    for piece in definitions:
        cleaned: str = piece.strip("\n")
        if not cleaned:
            continue
        identity = _definition_identity(piece)
        if identity is not None and identity in added_identities:
            continue
        index:Optional[int] = None if identity is None else places.get(identity)
        if index is None:
            # A piece with no definition shape of its own (a preprocessor block, say) can
            # only be appended: nothing identifies it, so nothing can match it.
            added.append(cleaned)
            if identity is not None:
                added_identities.add(identity)
            continue
        if identity == PROTOTYPE_MARKER:
            # The forward-declaration block is *unioned*, not replaced: a later call
            # renders only its own callees, and replacing the block would drop the ones
            # an earlier call declared -- calls the file still holds.
            from .prototypes import merge_prototype_blocks  # noqa: PLC0415 (import cycle)

            held:str = "\n\n".join(part.strip("\n") for part in groups[index][1]).strip("\n")
            unioned:Optional[str] = merge_prototype_blocks([held, cleaned])
            if unioned is not None and unioned != held:
                groups[index] = (identity, [unioned])
                changed = True
            continue
        held:str = "\n\n".join(part.strip("\n") for part in groups[index][1]).strip("\n")
        if held != cleaned:
            groups[index] = (identity, [cleaned])
            changed = True
    if added:
        groups.extend((_definition_identity(piece), [piece]) for piece in added)
        changed = True
    merged: List[str] = [piece for _identity, pieces in groups for piece in pieces]
    return merged, changed



def merge_source(existing: str, addition: str) -> str:
    """`existing` with every definition of `addition` applied, in place for a changed one.

    Python functions in one module have distinct names, so a definition cannot repeat; this
    only guards against a source being regenerated with pieces the file already holds
    (a kernel that was compiled, its file kept, and the same kernel compiled again) -- and
    against a piece the file holds that the generator now writes differently, which is what
    `_merge_definitions` refreshes rather than skips.
    """
    if not existing.strip():
        return addition
    chunks: List[str] = existing.split("\n\n")
    merged: List[str]
    merged, changed = _merge_definitions(chunks, addition.split("\n\n"))
    if not changed:
        return existing
    return "\n\n".join(piece.rstrip("\n") for piece in merged if piece.strip()) + "\n"



def assemble_source(header: List[str], definitions: List[str],
                    path: str = "") -> str:
    """One generated source: the file's guard, its header, then every definition.

    Preprocessor directives lead the header: they enable the types the definitions name
    (`#define QUAT_NEED_DOUBLE` ahead of the header that declares `quatd`), and the
    compiler reads them before anything that uses them. Within a block that mixes the two
    the directives move to its front for the same reason; a block of pure declarations
    stays where it is.

    The whole file is wrapped in its own guard, so a file reached from several others is
    read once. Each structure carries a guard of its own as well (see
    `structs.render_structures`), because two modules can name one layout -- one C type.
    """
    # The order the caller built is kept, not re-derived here: `_canonical_header` already
    # puts the `#define`/`#include` block where it belongs, and a structure declaration has
    # to come *before* an include whose file names it (a helper in another module takes a
    # `Ray` by pointer). Re-sorting by "a line starting with # leads" would move the
    # includes past the declarations and break exactly that.
    parts: List[str] = [piece.strip("\n") for piece in header if piece.strip()]
    parts.extend(piece.strip("\n") for piece in definitions if piece.strip())
    # A run of `#include` lines reads as one list, so the lines are packed together and one
    # blank line follows the last of them. Every other pair of pieces keeps its separator:
    # a structure declaration, a device function and the kernel each stand on their own.
    joined: List[str] = []
    for piece in parts:
        if joined and _is_include_only(joined[-1]) and _is_include_only(piece):
            joined[-1] = joined[-1] + "\n" + piece
            continue
        joined.append(piece)
    body: str = "\n\n".join(joined) + "\n"
    if not path:
        return body
    macro: str = guard_macro(path)
    return f"#ifndef {macro}\n#define {macro}\n\n{body}\n#endif\n"



def _split_source(source: str, path: str = "") -> Tuple[List[str], List[str]]:
    """A generated source as (header, definitions).

    The header is every piece that declares a type or a directive: structure typedefs,
    `#include` lines and preprocessor defines. The forward-declaration block is header
    material too -- it says nothing but `name;` declarations -- and it is kept as *one*
    piece so `_canonical_header` can union the blocks several calls wrote. Everything else
    is a function definition.
    """
    source = _strip_file_guard(source, path)
    file_macro: str = guard_macro(path) if path else ""
    pieces: List[str] = [piece for piece in source.split("\n\n") if piece.strip()]
    header: List[str] = []
    definitions: List[str] = []
    for piece in pieces:
        lines: List[str] = piece.splitlines()
        first: str = lines[0].strip() if lines else ""
        is_declaration: bool = (first.startswith("struct ")
                                or first.startswith("typedef struct")
                                or first.startswith("#")
                                or first == PROTOTYPE_MARKER)
        if not is_declaration:
            definitions.append(piece)
            continue
        # The file's own guard frames the whole file: it is not a declaration to carry
        # into the header of the next write.
        remainder: List[str] = [line for line in lines
                                if not _is_file_guard_line(line, file_macro)]
        block: str = "\n".join(remainder).strip("\n")
        if block:
            header.append(block)
    return header, definitions



def _canonical_header(header: List[str]) -> List[str]:
    """The header with one directive block, one copy of each other block.

    A generated file holds several kernels, each written on its own call, so the header
    collects one directive block per call: the block names the quaternion widths *that*
    kernel needs (`#define QUAT_NEED_DOUBLE` for a mixed quaternion operation). Left as
    they came, the file ends up with several blocks whose defines repeat; the reader's
    compiler then sees `QUAT_QUAL` defined again. Every directive line is folded into one
    block, and every `#include` and every declaration appears once.
    """
    defines: List[str] = []
    includes: List[str] = []
    declarations: List[str] = []
    seen_declarations: set = set()
    # The forward-declaration blocks several calls wrote: they are unioned into one at
    # the end, not deduped by exact text, because a later call may render fewer callees
    # than the file already declares (see `merge_prototype_blocks`).
    blocks: List[str] = []
    for piece in header:
        lines: List[str] = piece.splitlines()
        if lines and lines[0].strip() == PROTOTYPE_MARKER:
            blocks.append(piece.strip("\n"))
            continue
        directives: List[str] = [line for line in lines if line.lstrip().startswith("#")]
        remainder: List[str] = [line for line in lines if line not in directives]
        if remainder:
            # A block that mixes a directive with a declaration is kept whole: pulling the
            # guard out of a wrapped structure would leave the declaration bare, and
            # splitting a prelude (Metal's `#include`, `using namespace metal;` and its
            # macros) would leave the `using` line in a piece that `_split_source`
            # classifies as a function definition -- it was dropped on the second write
            # that way. Kept whole, the exact same text is written again and skipped by
            # the membership test below, which is the idempotence this relies on.
            block: str = piece.strip("\n")
            if block and block not in seen_declarations:
                seen_declarations.add(block)
                declarations.append(block)
            continue

        for line in directives:
            stripped: str = line.strip()
            if stripped.startswith("#include"):
                if stripped not in includes:
                    includes.append(stripped)
            elif stripped not in [item.strip() for item in defines]:
                defines.append(stripped)

    canonical: List[str] = []
    if defines:
        # `#define QUAT_QUAL` first, then the width switches: the header that follows tests
        # them in that order.
        qualifier: List[str] = [line for line in defines
                                if not line.startswith(QUAT_DEFINE_PREFIX)]
        widths: List[str] = [line for line in defines
                             if line.startswith(QUAT_DEFINE_PREFIX)]
        canonical.append("\n".join(qualifier + widths))
    # A *fixed* header declares types this file's own declarations may name -- a structure
    # member spelled `quatf` or `c64` -- so those includes come first. A module include is
    # different: the file it reaches can name a structure declared here (a helper in another
    # module takes a `Ray` by pointer), so it stays after the declarations.
    fixed_includes: List[str] = [line for line in includes
                                 if _is_fixed_header_include(line)]
    module_includes: List[str] = [line for line in includes
                                  if not _is_fixed_header_include(line)]
    canonical.extend(fixed_includes)
    canonical.extend(declarations)
    if blocks:
        # The forward declarations sit *before* the module includes: an included file's
        # definitions call functions this unit defines later, and C refuses the call as
        # an implicit declaration when it reads first. They follow the declarations
        # (typedefs), because a declaration may name a structure type.
        from .prototypes import merge_prototype_blocks  # noqa: PLC0415 (import cycle)

        merged_block: Optional[str] = merge_prototype_blocks(blocks)
        if merged_block:
            canonical.append(merged_block)
    canonical.extend(module_includes)
    return canonical
