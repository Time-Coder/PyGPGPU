# Finding the block a `with inline, kernel(...)` head opened, and reading its scope.
"""The two questions asked at capture time, before anything is translated.

`inline` is entered before the block manager exists, so it is the manager's `__enter__` that
records what the site is: the `ast.With` node, the caller's frame, and the caller's file.
It raises :class:`InlineBlock`; `inline.__exit__` catches it, suppresses it so the caller's
frame never runs a line of the body, and then launches.

The block's **data** is not written into the head. A `@kernel` body reads its values from
its parameters, a closure cell or a module global; a block has no parameter list, so the
names it reads from the enclosing frame become its parameters -- the caller writes nothing,
and `def add(a, b, out)` and a block that reads `a`, `b`, `out` reach the device the same
way. A module global needs no parameter, because the rebuilt function's globals dictionary
is the module's own.

Nothing here decides what the body *means*: that is the translator's job, unchanged. What
is refused is the spellings a `with` block cannot carry -- a one-line body, a top-level
`break`/`continue` (they would target the caller's loop), a `nonlocal` (there is no enclosing
scope to write), and an `await`/`yield` (this launch is synchronous).
"""
from __future__ import annotations

import ast
import sys
from types import FrameType
from typing import Dict, List, Optional, Tuple

from .rebuild import block_body_lines, block_name, block_text

from ..translator.dispatch_domain import _is_magic_loop_stmt


# The device builtins that mean "this body reads a work item of its own": a body that calls
# one is a gid-style body, which an inline block refuses (see `_refuse_a_gid_style_body`).
_WORK_ITEM_QUERIES: frozenset = frozenset({
    "get_global_id", "get_local_id", "get_group_id", "get_local_size", "get_global_size",
    "get_num_groups", "get_work_dim", "get_global_offset",
})


class InlineBlock(Exception):
    """Raised by the block manager's `__enter__`, caught by `inline.__exit__`.

    It does not mean "something went wrong": it is how a block says "I am here, do not run
    me". The exception carries everything the launch needs, because it is the only thing
    that crosses from the manager (which knows the options) to `inline` (which knows the
    block).
    """

    def __init__(self, spec:object, node:ast.With, frame:FrameType,
                 filename:str, names:List[str])->None:
        super().__init__(
            "an inline kernel block is launched when its `with` statement exits, not "
            "entered: write it as `with inline, kernel(...):`")
        self.spec:object = spec
        self.node:ast.With = node
        self.frame:FrameType = frame
        self.filename:str = filename
        # The captured parameter names, in launch order (the receiver first, then the order
        # the body first reads each one).
        self.names:List[str] = names
        self.consumed:bool = False


def _caller_frame()->FrameType:
    """The frame that wrote the `with` statement, skipping this package's own frames."""
    frame:Optional[FrameType] = sys._getframe(2)
    while frame is not None and frame.f_globals.get("__name__", "").startswith("pygpgpu"):
        frame = frame.f_back
    if frame is None:
        raise SyntaxError("`inline` was entered with no caller frame to read the block from")
    return frame


def _source_lines(filename:str)->List[str]:
    """The caller file's lines, which the generated text is assembled from.

    A file that cannot be read (``python -c``, stdin, a frozen build) is refused rather than
    served a guess: the block's own text is the only thing that says what it means.
    """
    try:
        with open(filename, encoding="utf-8") as handle:
            return handle.read().splitlines()
    except OSError as error:
        raise SyntaxError(
            f"an inline kernel reads the block from its source file, but {filename!r} "
            f"cannot be read ({error.strerror}). Run it from a .py file rather than "
            f"`python -c`/stdin, or write the kernel with @kernel") from None


def _is_the_inline_object(node:ast.expr, frame:FrameType, inline_object:object)->bool:
    """True when this item's expression is the `inline` singleton, not some other value."""
    if not isinstance(node, ast.Name):
        return False
    value:object = frame.f_locals.get(node.id, frame.f_globals.get(node.id))
    return value is inline_object


def _find_with(frame:FrameType, inline_object:object)->Tuple[ast.With, List[str], str]:
    """The `ast.With` node of the statement being entered, the file's lines, and its name.

    The line number is the identity: `frame.f_lineno` inside the manager's `__enter__` is
    always the statement's *first* line, so a multi-line head still matches, and a `with` is
    compound, so no two of them can start on one line. The node is then chosen by span,
    because an enclosing `with` may span the same line -- the innermost one is the statement
    being entered.
    """
    filename:str = frame.f_code.co_filename
    source_lines:List[str] = _source_lines(filename)
    line:int = frame.f_lineno
    tree:ast.Module = ast.parse("\n".join(source_lines), filename)
    candidates:List[ast.With] = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and node.lineno <= line <= (node.end_lineno or node.lineno)
        and isinstance(node.items[-1].context_expr, ast.Call)
        and _is_the_inline_object(node.items[0].context_expr, frame, inline_object)]
    if not candidates:
        raise SyntaxError(
            "`kernel(...)` was entered as a context manager, which only an inline kernel "
            "does: it must be the last item of `with inline, kernel(...):`")
    return max(candidates, key=lambda found: found.lineno), source_lines, filename


def _refuse_a_one_line_body(node:ast.With)->None:
    """A one-line body (`with ...: out[i] = x`) has no text of its own to rebuild from."""
    if node.body[0].lineno == node.lineno:
        raise SyntaxError(
            "an inline kernel block has to be written on its own lines, below the `with` "
            "head: `with inline, kernel(...):` and then the body indented under it")


def _inside_a_block_loop(body:List[ast.stmt], target:ast.AST)->bool:
    """True when `target` sits inside a loop written *in the block*."""
    for statement in body:
        if isinstance(statement, (ast.For, ast.While, ast.AsyncFor)):
            for child in ast.walk(statement):
                if child is target:
                    return True
    return False


def _refuse_a_top_level_jump(node:ast.With)->None:
    """A top-level `break`/`continue` targets the caller's loop, which the block never enters."""
    for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
        if not isinstance(child, (ast.Break, ast.Continue)):
            continue
        if _inside_a_block_loop(node.body, child):
            continue
        keyword:str = "break" if isinstance(child, ast.Break) else "continue"
        raise SyntaxError(
            f"a top-level `{keyword}` cannot appear in an inline kernel block: it would "
            f"target a loop of the enclosing function, and the block does not run there. "
            f"Put it inside the block's own `for` loop, where it keeps its device meaning")


def _refuse_a_nonlocal(node:ast.With)->None:
    """The rebuilt block is module-level, so `nonlocal` has no enclosing scope to write."""
    for child in ast.walk(node):
        if isinstance(child, ast.Nonlocal):
            names:str = ", ".join(child.names)
            raise SyntaxError(
                f"`nonlocal {names}` cannot appear in an inline kernel block: the block is "
                f"translated as a kernel body, which has no enclosing scope to write. "
                f"Capture a one-element array and write through it instead")


def _refuse_await_and_yield(node:ast.With)->None:
    """v1 launches synchronously; an `await`/`yield` would need a second execution model."""
    for child in ast.walk(node):
        if isinstance(child, (ast.Await, ast.Yield, ast.YieldFrom)):
            keyword:str = "await" if isinstance(child, ast.Await) else "yield"
            raise SyntaxError(
                f"`{keyword}` cannot appear in an inline kernel block: the block is "
                f"launched synchronously, so it is neither a coroutine nor a generator")


def _magic_loops(node:ast.With)->List[ast.For]:
    """The block's magic loops: its top-level `for i in range(<array>.shape[0])` dispatch.

    No parameter set is passed, and that is the correct answer here rather than an omission:
    an inline block has no parameter list of its own. Its body reads the *enclosing* scope,
    so a `range(n)` bound there names a local or a closure cell -- a value the source decides,
    not a count the call hands over -- and promoting it would turn an ordinary serial loop
    into a grid. The shape spelling (`range(out.shape[0])`) is what a block states its domain
    with, which is also what its own error message asks for.
    """
    return [statement for statement in node.body
            if isinstance(statement, ast.For)
            and _is_magic_loop_stmt(statement, frozenset())]


def _launches_a_child_kernel(node:ast.With)->bool:
    """True when the block's body hands the work to a kernel it launches from the device.

    A body that launches a child with a bracket (`child[(n,)](...)`) does not need a magic loop
    of its own: the *child* is what divides the work, and the block is the single work item
    that launches it. This is the device-side launch CUDA calls dynamic parallelism, and it is
    the reason the two spellings coexist -- the loop states a domain, a launch delegates one.
    """
    return any(
        isinstance(child, ast.Call) and isinstance(child.func, ast.Subscript)
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])))


def _refuse_a_gid_style_body(node:ast.With)->None:
    """An inline block must state its domain with a magic loop, not with `get_global_id`.

    A block is written at the launch site, so the extent it iterates is already in scope --
    a magic loop says the domain in the body itself, and the work size follows from the
    array it names. A gid-style body states the domain *outside* the body, and a block has
    nowhere to write that: `domain=` on the head would be a second, silent spelling of the
    thing the loop already says. So the body has to carry its own loop, and a block that
    reads `get_global_id` is refused with the loop it should write instead.

    The one other way a body states where its work goes is by launching a child kernel
    (`child[(n,)](...)`): then the child's launch is the work size, and the block is the
    single work item that starts it. That is a legitimate body, so it is let through.
    """
    if _magic_loops(node) or _launches_a_child_kernel(node):
        return
    uses_gid:bool = any(
        isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        and child.func.id in _WORK_ITEM_QUERIES
        for child in ast.walk(node))
    if not uses_gid:
        raise SyntaxError(
            "an inline kernel block needs a magic loop to state its work size -- "
            "`for i in range(<array>.shape[0]):` -- because a block is launched where it is "
            "written and has no other place to declare the domain. Put the body inside that "
            "loop, launch a child kernel (`child[(n,)](...)`) to divide the work instead, or "
            "write it as a @kernel function with `domain=`")
    raise SyntaxError(
        "an inline kernel block does not support a gid-style body: `get_global_id` reads a "
        "work item, but the work size has to come from the body itself. Write the domain as a "
        "magic loop instead -- `for i in range(<array>.shape[0]):` -- and index with `i`; "
        "the loop bound IS the work size. A gid-style body is a @kernel function's spelling, "
        "where `@kernel(domain=\"...\")` can name the array beside the body")


def _bound_in_the_block(body:List[ast.stmt])->set:
    """The names the block itself binds, which are therefore not captured from the frame."""
    bound:set = set()
    for child in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(child, ast.Name) and not isinstance(child.ctx, ast.Load):
            bound.add(child.id)
        elif isinstance(child, ast.arg):
            bound.add(child.arg)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            bound.update(child.names)
    return bound


def _read_order(body:List[ast.stmt])->List[str]:
    """The names the block reads, in source order (position of first read).

    `ast.walk` is breadth-first, which for a two-statement body would interleave the two
    statements; sorting the reads by position restores the order a reader sees, so the
    generated parameter list reads top-to-bottom.
    """
    reads:List[Tuple[int, int, str]] = []
    for child in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            reads.append((child.lineno, child.col_offset, child.id))
    order:List[str] = []
    for _line, _column, name in sorted(reads):
        if name not in order:
            order.append(name)
    return order


def captured_names(node:ast.With, frame:FrameType)->List[str]:
    """The frame names the block reads: its parameter list, in launch order.

    A name the block resolves through the module globals needs no parameter, because the
    rebuilt function's globals dictionary is the module's own. What is captured is therefore
    a name of the **enclosing function's locals** -- and that check comes first, because a
    local shadowing a module global is exactly the name the block must see the local value
    of; testing the globals first would silently pass the global's value instead.

    A method's receiver is hoisted to the front, which is where a kernel's receiver parameter
    belongs: `PyKernel` reads the first parameter's name to decide what travels separately.
    """
    bound:set = _bound_in_the_block(node.body)
    names:List[str] = []
    for name in _read_order(node.body):
        if name in bound:
            continue
        if name in frame.f_locals:
            names.append(name)
    for receiver in ("self", "cls"):
        if receiver in names:
            names.remove(receiver)
            names.insert(0, receiver)
    return names


def _refuse_a_domain_option(spec:object)->None:
    """`domain=` is a decorator's spelling; a block states its domain with the loop itself.

    An inline block is written where it is launched, so the array whose shape gives the work
    size is already in scope -- the magic loop names it directly. Carrying `domain=` as well
    would be a second, silent spelling of that same fact, so it is refused and the body's own
    loop is named as the fix.
    """
    domain:object = getattr(spec, "domain", None)
    if domain is None:
        return
    raise SyntaxError(
        "an inline kernel block does not take `domain=`: the block is written where it is "
        "launched, so its body states the domain itself with a magic loop -- "
        "`for i in range(<array>.shape[0]):` -- and the work size follows from the array that "
        "loop names. `@kernel(domain=\"...\")` is the decorator's spelling, used where the body "
        "is written away from the array it runs over")


def _refuse_a_block_inside_a_kernel_body()->None:
    """A block may not be written inside a `@kernel` body: that body is device code.

    The two backends used to disagree here, and the disagreement was silent. A device backend
    translates the body and refuses the `with` (`unsupported statement: With`), because a
    device has no `with`; the interpreter backend *runs* the body, so the block launched
    inside the outer loop and the outer kernel answered a different number. One spelling has
    to mean one thing everywhere, so it is refused where it can be caught on both: while the
    `with` head is entered, which happens inside the kernel's own execution.

    The python backend installs its launch frame for exactly the duration of a kernel call
    (`host_runtime.set_host_frame`), which is what makes that observable. A device backend
    never runs the body, so it never reaches here -- its translator reports the same thing in
    its own words, and both refuse rather than one computing.
    """
    from ..host_runtime import get_host_frame

    if get_host_frame() is not None:
        raise SyntaxError(
            "an inline kernel block cannot be written inside a @kernel body: that body is "
            "already device code, so it has no `with` statement and no second launch. Write "
            "the block at the launch site (in ordinary python, outside every @kernel), or "
            "make this a device helper the kernel calls from inside its loop")


def capture(spec:object, inline_object:object)->InlineBlock:
    """Turn the caller's `with` statement into the block the launch will translate.

    Everything refused here is refused *before* translation, because the alternative is a
    translator message about syntax the user never believed they wrote.
    """
    _refuse_a_block_inside_a_kernel_body()
    _refuse_a_domain_option(spec)
    frame:FrameType = _caller_frame()
    node, _source, filename = _find_with(frame, inline_object)
    _refuse_a_one_line_body(node)
    _refuse_a_top_level_jump(node)
    _refuse_a_nonlocal(node)
    _refuse_await_and_yield(node)
    _refuse_a_gid_style_body(node)
    return InlineBlock(spec, node, frame, filename, captured_names(node, frame))


def text_of(site:InlineBlock, source_lines:Optional[List[str]]=None)->str:
    """The generated `def` text for one captured block, with its captures as parameters."""
    lines:List[str] = source_lines if source_lines is not None else _source_lines(site.filename)
    return block_text(lines, site.node, site.names)


def name_of(site:InlineBlock)->str:
    """The generated function's name for one block site."""
    return block_name(site.node.lineno)


def body_lines(site:InlineBlock)->List[str]:
    """The block's own lines, dedented: the function body the text is assembled from."""
    return block_body_lines(_source_lines(site.filename), site.node)


def arguments_of(entry:object, site:InlineBlock)->Tuple[object, ...]:
    """This call's captured values, read live from the frame in launch order."""
    locals_now:Dict[str, object] = site.frame.f_locals
    return tuple(locals_now[name] for name in entry.names)
