# The forward-declaration block one generated file writes for callees it does not define.
from __future__ import annotations
from .guards import PROTOTYPE_MARKER
from .merge import _definition_identity
from typing import Dict, List, Optional, Set


def prototype_block(prototypes: List[str]) -> str:
    """One guarded block of forward declarations, as a single definition-shaped piece.

    C requires a function to be declared before its caller, and one generated file holds
    several kernels written on separate calls. A callee that lives in *another* generated
    file is reached with an `#include`, but a file that is only written later still has a
    call in it before the definition exists: the compiler reads the caller first and
    refuses (`implicit declaration of function ... is invalid in OpenCL`), and the later
    definition then answers "conflicting types" against that implicit one. Declaring the
    callee's header line here is what makes the unit self-contained regardless of the
    order two modules' kernels were first compiled in.

    The block carries only declarations -- each line is the callee's rendered signature
    followed by `;` -- so it is safe on every backend this package generates for. The
    marker is part of the block, not a second piece: a piece with no identity is appended
    rather than matched (see `_merge_definitions`), and the block has to be matched.
    """
    lines: List[str] = [PROTOTYPE_MARKER]
    lines.extend(prototype for prototype in prototypes if prototype.strip())
    return "\n".join(lines)


def prototype_prototypes(block: str) -> List[str]:
    """The declaration lines a prototype block holds, without the marker."""
    return [line.strip() for line in block.splitlines()
            if line.strip() and line.strip() != PROTOTYPE_MARKER]


def merge_prototype_blocks(blocks: List[str]) -> Optional[str]:
    """One block holding the union of the blocks given, or None when there is none.

    A file is rewritten on every kernel call of its module, and each call renders only
    *its* callees: unioning is what keeps an earlier call's declarations from being
    dropped when a later one carries fewer. The union is keyed by the declaration's own
    text, and the first spelling of each wins -- so a later re-render that spells the
    same callee identically cannot flip the block's order around.
    """
    declarations: List[str] = []
    seen: Set[str] = set()
    for block in blocks:
        for prototype in prototype_prototypes(block):
            if prototype in seen:
                continue
            seen.add(prototype)
            declarations.append(prototype)
    return prototype_block(declarations) if declarations else None


def _is_declaration_line(line: str) -> bool:
    """Whether one line reads as a C function signature (`<qualifiers> <type> <name>(...)`).

    This is the shape a definition head has, and the same shape a forward declaration
    needs once a `;` is appended. Statements are told apart structurally: a body line
    carries a call on its own line only when the renderer split one there, which it does
    not -- but the `=`, control-word and semicolon tests keep a statement from passing
    even if it did.
    """
    stripped: str = line.strip()
    if not stripped or stripped.endswith(";") or stripped.startswith("#"):
        return False
    if "(" not in stripped:
        return False
    head: str = stripped.split("(", 1)[0].strip()
    if not head or "=" in head:
        return False
    first: str = head.split()[0]
    if first in ("if", "else", "for", "while", "do", "switch", "case", "return",
                 "break", "continue", "struct", "typedef"):
        return False
    # `_definition_identity` answers the line itself for exactly this shape; None here
    # means the line is body text (a `{`, a closing brace, a bare word).
    return _definition_identity(stripped) is not None


def _prototypes_of(codes: List[str]) -> List[str]:
    """The forward declarations a set of rendered functions needs from its callers.

    `render_device` writes the signature as each function's first line, so that line is
    the declaration a caller's file may need before the definition exists. Every function
    contributes one: the caller is a translation unit that includes (or will include)
    this code's file, and C reads its own text first -- whether the definition has been
    written beside it yet or not.
    """
    prototypes: List[str] = []
    seen: Set[str] = set()
    for code in codes:
        lines: List[str] = code.splitlines()
        if not lines or not _is_declaration_line(lines[0]):
            continue
        prototype: str = lines[0].strip() + ";"
        if prototype in seen:
            continue
        seen.add(prototype)
        prototypes.append(prototype)
    return prototypes


def compilation_prototypes(entries: List[Dict[str, object]], pieces: List[str]) -> List[str]:
    """The forward declarations one compilation's whole callee set needs.

    Every file written for the compilation carries the same block, declaring every device
    function the compilation reached: a cache entry (own or foreign) and a fixed helper
    the translator emitted (`device_helpers`, a nested helper or kernel). A foreign
    definition may call either -- the receiver-struct helper it was translated against,
    or the floor-division helper a `//` in its body spelled -- and both are defined in
    *another* file, later in the unit or pasted in behind the declarations.
    """
    codes: List[str] = [entry.get("code") or "" for entry in entries]
    codes.extend(piece for piece in pieces if piece)
    return _prototypes_of(codes)
