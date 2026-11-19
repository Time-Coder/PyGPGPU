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


def _callee_name(head_line: str) -> str:
    """The identifier one definition head line declares: the word before its `(`."""
    head: str = head_line.split("(", 1)[0].strip()
    return head.split()[-1] if head else ""


def narrow_prototype_block(block: str, definitions: List[str]) -> Optional[str]:
    """The declaration lines of `block` that the given definitions actually call.

    A file's block names only what its own definitions reach (see
    `compilation_prototypes`); the merge unions the block a later call renders with
    the one the file already holds, and this is where the union is kept honest: a
    declaration whose callee no definition in the file names anymore -- the one a
    package-side regeneration stopped spelling, or an earlier package version stacked
    the whole unit's calls into every file -- is dropped instead of surviving forever.
    `definitions` is the merged file's *definition* pieces -- the block itself is
    excluded by the caller, because a declaration names its own callee and would
    otherwise always match itself. The match reads the bodies, never the head lines
    (a head line spells the function's own name, which is a definition and not a call
    of it). A callee the file also defines stays declared when a body calls it -- the
    emitter orders callees before callers, but a call cycle has no leaf, and the
    first definition needs the line. An empty answer means the file calls nothing,
    and the block is removed altogether.
    """
    declarations: List[str] = prototype_prototypes(block)
    if not declarations:
        return None
    body: str = "\n".join("\n".join(code.splitlines()[1:]) for code in definitions if code)
    kept: List[str] = []
    for declaration in declarations:
        name: str = _callee_name(declaration.rstrip(";"))
        if not name or name not in body:
            continue
        kept.append(declaration)
    return prototype_block(kept) if kept else None


def _called_names(codes: List[str], known: Set[str]) -> Set[str]:
    """The subset of the known callee names that the given code texts call.

    `known` is the set of generated C names the compilation reached (one per cache
    entry and fixed helper). Only the code *below* a definition's head line is read:
    a head line spells the function's own name, which is a definition, not a call of
    it -- matching it would make every function declare itself. The body is where
    calls live, and a generated name is a distinct identifier, so a containment test
    over it cannot hit a longer name's substring.
    """
    called: Set[str] = set()
    for code in codes:
        if not code:
            continue
        lines: List[str] = code.splitlines()
        body: str = "\n".join(lines[1:])
        for name in known:
            if name in body:
                called.add(name)
    return called


def compilation_prototypes(entries: List[Dict[str, object]], pieces: List[str],
                           own_codes: Optional[List[str]] = None) -> List[str]:
    """The forward declarations one file's own code needs from this compilation.

    `entries` and `pieces` describe the compilation's whole callee set -- every cache
    entry (own or foreign) and fixed helper the translator emitted -- and each one's
    definition head line is the declaration a call can name. `own_codes` are the
    *rendered definitions this file carries*: the kernel's own pieces for the kernel's
    file, a foreign module's entries for its file. A file's block declares the callees
    its own definitions *call* -- the match reads the bodies, never the head lines (a
    head line spells the function's own name, which is a definition and not a call of
    it). The kernel's file names the helpers its pieces reach, a foreign file names
    the helpers and third-module functions its entries reach, and a file whose code
    calls nothing carries no block at all. A name the same file also defines stays
    declared when a call spells it: the emitter orders callees before callers, but a
    call cycle (two mutually recursive helpers) has no leaf, and the first definition
    needs the line. An included file's definitions still call functions the kernel's
    file defines later, and those calls name exactly the lines the block carries.

    (One block per compilation used to declare *every* callee the whole compilation
    reached, in every file it wrote -- mostly right because any file could call any
    helper, but it stacked declarations a file never calls: a header-only file declared
    the kernel that includes it, and every file declared the whole unit's kernel set.)
    """
    all_codes: List[str] = [entry.get("code") or "" for entry in entries]
    all_codes.extend(piece for piece in pieces if piece)
    heads: Dict[str, str] = {}
    for code in all_codes:
        if not code:
            continue
        first_line: str = code.splitlines()[0].strip()
        if first_line and _is_declaration_line(first_line):
            heads[_callee_name(first_line)] = first_line
    own: List[str] = (own_codes if own_codes is not None else all_codes)
    referenced: Set[str] = _called_names(own, set(heads))
    declared: List[str] = []
    seen: Set[str] = set()
    for name, head in heads.items():
        # A name this file defines is declared too when its own text references it:
        # the definitions are emitted callee-before-caller, but a call CYCLE (two
        # mutually recursive helpers) has no leaf to start from, and the first
        # definition reads a callee C has not seen. Declaring a function the same
        # file also defines is legal C and costs one line.
        if name not in referenced:
            continue
        prototype: str = head + ";"
        if prototype in seen:
            continue
        seen.add(prototype)
        declared.append(prototype)
    return declared
