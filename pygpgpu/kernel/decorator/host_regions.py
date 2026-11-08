# The host-side statements of a body, and their values at launch time.
from __future__ import annotations
from .slots import _kernel_scalar_argument
import ast
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
from ..python_backend import analyse
from ..translator import (
    _assigned_name_of,
    infer_ctype,
)



def _host_regions(func:object)->List[object]:
    """The statements a kernel body runs on the host, before and between its magic loops."""
    from ..translator import _parse_kernel_source, _outside_loops, get_dedented_source

    try:
        tree:object = _parse_kernel_source(get_dedented_source(func))
    except SyntaxError:
        return []
    if not (tree.body and isinstance(tree.body[0], ast.ImportFrom)):
        pass
    body:List[object] = tree.body[0].body if tree.body else []
    return list(_outside_loops(body))


def _host_scalar_values(func:object, names:List[str], args:Optional[Tuple[object, ...]]=None,
                        instance:object=None)->Dict[str, object]:
    """Evaluate the initializers a carried scalar needs, on the host.

    A carried scalar (`total = 0` above the magic loop, `total = total + x[i]` inside it) is
    one location for the whole launch, so its starting value has to be a real python object
    -- the same one the interpreter would have made. The statements above the loop are run
    here in body order, sharing one namespace so a later initializer can read an earlier
    name, exactly as python would.
    """

    values:Dict[str, object] = {}
    if not names:
        return values
    globalns:Dict[str, object] = dict(func.__globals__)
    localns:Dict[str, object] = {}
    from ..decorator import _positional_argument_names
    for arg_name, value in zip(_positional_argument_names(func), args or ()):
        localns[arg_name] = value
    from ..decorator import _receiver_name
    receiver:str = _receiver_name(func)
    if receiver and instance is not None:
        localns[receiver] = instance
    for name, cell in zip(func.__code__.co_freevars, func.__closure__ or ()):
        try:
            localns[name] = cell.cell_contents
        except ValueError:
            pass
    # Every statement that runs on the host, in body order: a gap statement is host code even
    # when the translator cannot lift it for a creation's sake, which is why the whole
    # host region is walked rather than only the statements a creation needs.
    statements = _host_regions(func)
    for statement in statements:
        target:Optional[str] = _assigned_name_of(statement)
        if target is None:
            continue
        try:
            localns[target] = eval(  # noqa: S307 - the kernel source itself
                compile(ast.Expression(body=statement.value), "<kernel scalar>", "eval"),
                globalns, localns)
        except Exception:
            continue
    for name in names:
        if name in localns:
            values[name] = localns[name]
    return values


def _reads_name(node:object, name:str)->bool:
    """Whether an expression reads the given plain name."""
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _is_array_pointer_assignment(statement:object)->bool:
    """True for `local = <array>` or `local = <array>[<slice>]`: both bind a device pointer.

    The right-hand side has to be a *bare* name or receiver attribute, or a slice of one:
    anything else is an expression the host may well be the one to evaluate.
    """
    if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)):
        return False
    value:object = statement.value
    if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Slice):
        return True
    if isinstance(value, ast.Name):
        return True
    return isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)


def _host_region_values(func:object,
                        args:Optional[Tuple[object, ...]]=None,
                        instance:object=None,
                        created_names:Optional[Set[str]]=None
                        )->Tuple[Dict[str, object], List[int]]:
    """Run every host-region statement, once, and report what it bound.

    A kernel body's statements above its first magic loop, and between two of them, are
    ordinary python: the interpreter runs them **once**, as the body reads -- not once per
    work item. That is the rule the rest of this package already follows for the tail
    (PyKernel._run_host_region); the region above and between the loops is host code for
    the same reason, so it is executed here through the function's own namespace: its
    globals, its positional arguments, its receiver and its closure cells.

    The second result is the list of *line numbers of the statements that ran*: those are
    host work, and the translator emits no device code for them (see
    `FunctionTranslator.host_statement_lines`). A statement the host could not run is left
    out of it, and stays device code -- that is how a device-only builtin such as
    `zeros_per_group(...)` above a loop keeps working.

    The body's tail -- the statements after the last magic loop -- is host work too (the
    host runs it once, after the launch; see PyKernel._run_host_region), so it is executed
    and reported here as well. A `return` in the tail is skipped: it answers the launch's
    value, and evaluating it twice would run its expression twice.

    A statement that assigns an **outer-scope** name (`nonlocal held; held = fresh`) is
    left out: that name travels through its own device slot and the *device* writes it
    (see _bind_outer_variables). Running it here would overwrite the slot's initial value
    from the host, and the value the kernel wrote would be lost.

    A **creation** (`mid = np.zeros(4)`) is left out too, when `created_names` says the
    build already produced that object: `_host_created_values` decides whether it travels
    as a device object, and building a second one here would shadow the launch's copy.
    """
    from ..translator import (_is_magic_loop_stmt, _magic_loop_end,
                             _outside_loops, get_dedented_source, _parse_kernel_source,
                             _outer_scope_names)

    values:Dict[str, object] = {}
    executed:List[int] = []
    try:
        tree:object = _parse_kernel_source(get_dedented_source(func))
    except SyntaxError:
        return values, executed
    if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
        return values, executed
    body:List[object] = list(tree.body[0].body)
    # A body with no magic loop is a *gid-style* body (`@kernel(domain="x")` or a bracket):
    # every one of its top-level statements is per-work-item code, so none of them is host
    # work. Running `atomic_add(...)` there would execute it once on the host (it is a
    # device builtin and refuses) and then once per work item on the device -- the atomic
    # landed twice.
    if not any(_is_magic_loop_stmt(statement)
               for statement in body if isinstance(statement, ast.For)):
        return values, executed
    # `local = <array>` is a pointer assignment, not host work: the device binds a second
    # name to the same buffer, and the array it names is a kernel argument (so its extents
    # are launch parameters). Running it here made the array look like a host value, and the
    # generated signature then lost the argument the shape came from.
    body = [statement for statement in body
            if not _is_array_pointer_assignment(statement)]
    outer_names:Set[str] = _outer_scope_names(func)
    built:Set[str] = set(created_names or ())
    statements:List[object] = list(_outside_loops(body))
    statements = [statement for statement in statements
                  if _assigned_name_of(statement) not in outer_names
                  and _assigned_name_of(statement) not in built]
    # The tail after the last magic loop, with any `return` left out: it is host work, and a
    # name it binds (`rest = helper()` after the loop) has to reach the host that runs it.
    tail:List[object] = [statement for statement
                         in body[_magic_loop_end(body):]
                         if not isinstance(statement, ast.Return)
                         and _assigned_name_of(statement) not in outer_names
                         and _assigned_name_of(statement) not in built]
    statements.extend(tail)
    if not statements:
        return values, executed

    globalns:Dict[str, object] = dict(func.__globals__)
    localns:Dict[str, object] = {}
    for arg_name, value in zip(_positional_argument_names(func), args or ()):
        localns[arg_name] = value
    receiver:str = _receiver_name(func)
    if receiver and instance is not None:
        localns[receiver] = instance
    for name, cell in zip(func.__code__.co_freevars, func.__closure__ or ()):
        try:
            localns[name] = cell.cell_contents
        except ValueError:
            pass

    for statement in statements:
        # One `exec` per statement, so an `import` binds its module and a nested block runs
        # its own statements; the region is a run of statements in body order sharing one
        # namespace, exactly like a function body.
        try:
            module:ast.Module = ast.Module(body=[statement], type_ignores=[])
            exec(compile(module, "<kernel host region>", "exec"),  # noqa: S102 - the kernel source itself
                 globalns, localns)
        except Exception:
            # The host cannot run it, so it is not host work: the translator keeps it and
            # reports its own refusal if it is not device code either.
            continue
        lineno:int = int(getattr(statement, "lineno", 0) or 0)
        if lineno:
            executed.append(lineno)
        target:Optional[str] = _assigned_name_of(statement)
        if target is not None and target in localns:
            values[target] = localns[target]
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
                and isinstance(statement.targets[0], ast.Tuple):
            # `lo, hi = bounds(1)`: each unpacked name is a value of its own.
            for element in statement.targets[0].elts:
                if isinstance(element, ast.Name) and element.id in localns:
                    values[element.id] = localns[element.id]
    return values, executed


def _host_created_values(func:object,
                         args:Optional[Tuple[object, ...]]=None,
                         instance:object=None,
                         required_created:Optional[List[str]]=None
                         )->Tuple[Dict[str, object], List[str]]:
    """Build the objects a kernel creates above its magic loop.

    The statements are ordinary python and are run as such, in the function's own
    namespace, so `np.zeros(x.shape[0], dtype=x.dtype)` means exactly what it means
    outside a kernel -- including reading the arguments the caller passed (`args`), which
    is why this runs at call time, not at decoration time. `instance` is the receiver when
    the kernel is a method, so a creation can read `self` (`np.zeros((self.height, ...))`).
    Only the statements the translator recognises as pure creation are executed; everything
    else stays device code.
    """
    from ..translator import (_creation_assignments,
                            _creation_target_of, _is_creation_expression)

    values:Dict[str, object] = {}
    names:List[str] = []
    try:
        assignments:List[object] = _creation_assignments(func)
    except SyntaxError:
        return values, names
    if not assignments:
        return values, names

    globalns:Dict[str, object] = dict(func.__globals__)
    localns:Dict[str, object] = {}
    for arg_name, value in zip(_positional_argument_names(func), args or ()):
        localns[arg_name] = value
    receiver:str = _receiver_name(func)
    if receiver and instance is not None:
        localns[receiver] = instance
    for name, cell in zip(func.__code__.co_freevars, func.__closure__ or ()):
        try:
            localns[name] = cell.cell_contents
        except ValueError:
            pass
    wanted:set = set(required_created or [])
    for assignment in assignments:
        target:Optional[str] = _assigned_name_of(assignment)
        if target is None:
            continue
        if target not in wanted and _creation_target_of(assignment) is None:
            # A plain assignment is lifted only when a creation needs it (or a creation
            # it feeds was asked for by name).
            if not (target in localns and any(
                    _reads_name(other.value, target) for other in assignments)):
                continue
        try:
            value:object = eval(  # noqa: S307 - the kernel source itself
                compile(ast.Expression(body=assignment.value), "<kernel creation>", "eval"),
                globalns, localns)
        except Exception:
            continue  # the translator reports the real problem, with a source position
        localns[target] = value
        # Only a *creation* becomes a kernel parameter: its object has to reach the device.
        # A plain assignment lifted for the creation's sake stays in the namespace alone --
        # the work items compute it themselves, so shipping it would add an argument nothing
        # reads (and shift every later launch argument by one).
        if _creation_target_of(assignment) is None and target not in wanted:
            continue
        if target in wanted or (_creation_target_of(assignment) is not None
                                and _is_creation_expression(assignment.value, func)):
            if _creation_target_of(assignment) is None:
                # A plain assignment the caller asked for feeds a creation below the loop;
                # it stays out of the launch values, it only fills the namespace.
                localns.setdefault(target, value)
                continue
            values[target] = value
            names.append(target)
    return values, names


def _receiver_name(func:object)->str:
    """The name of a method's receiver parameter (`self` / `cls`), or "" for a function."""
    code:object = getattr(func, "__code__", None)
    if code is None or not code.co_varnames:
        return ""
    first:str = code.co_varnames[0]
    return first if first in ("self", "cls") else ""


def _declares_a_magic_loop(func:Callable)->bool:
    """True when the kernel body carries a magic loop, i.e. it fixes its own domain.

    The body scan the interpreter already performs answers this without translating: it
    records the loop's form and the array it iterates.
    """
    return analyse(func, _receiver_name(func)).magic_form is not None


def _positional_argument_names(func:object)->List[str]:
    """The python parameter names of a kernel, skipping the `self` parameter if present."""
    code:object = getattr(func, "__code__", None)
    if code is None:
        return []
    names:List[str] = list(code.co_varnames[:code.co_argcount])
    if names and names[0] in ("self", "cls"):
        return names[1:]
    return names


def _created_arguments(created:Dict[str, object])->List[object]:
    """The launch arguments for the objects a kernel body created, in declaration order."""
    return [_kernel_scalar_argument(value, infer_ctype(value)) for value in created.values()]


def _called_modules(path:str, foreign_entries:Dict[str, List[Dict[str, object]]]) -> set:
    """The other generated files one module's definitions call functions from.

    A module's own file holds its definitions; a definition that names another module's
    device function needs that file, and C requires the callee before its caller. The
    reference is matched textually against the callee's generated name, the way
    `_dependency_order` orders the definitions themselves.
    """
    own_code:str = "\n\n".join(entry["code"] for entry in foreign_entries.get(path, ())
                               if entry["code"])
    called:set = set()
    for other, entries in foreign_entries.items():
        if other == path:
            continue
        for entry in entries:
            name:str = entry.get("c_name") or ""
            if name and name in own_code:
                called.add(other)
                break
    return called


def _dependency_order(device_cache:Dict[object, object])->List[Dict[str, object]]:
    """Order cached device functions so each callee precedes its callers.

    The cache is populated depth-first, so a nested helper (e.g. `clamp01` called by
    `weight`) is completed after the helper that calls it. Emitting in insertion order
    would then reference a function C has not seen yet. Names are matched textually
    against each entry's generated code.
    """
    entries:List[Dict[str, object]] = [e for e in device_cache.values() if e.get("status") == "done"]
    if len(entries) <= 1:
        return entries

    # name -> entry, for the callee lookup
    by_name:Dict[str, Dict[str, object]] = {e["c_name"]: e for e in entries}
    ordered:List[Dict[str, object]] = []
    visiting:set = set()
    done:set = set()

    def visit(entry:Dict[str, object]) -> None:
        name:str = entry["c_name"]
        if name in done or name in visiting:
            # `visiting` breaks cycles: mutual recursion is rejected elsewhere anyway.
            return
        visiting.add(name)
        for other_name in by_name:
            if other_name != name and other_name in entry["code"]:
                visit(by_name[other_name])
        visiting.discard(name)
        done.add(name)
        ordered.append(entry)
    for entry in entries:
        visit(entry)
    return ordered
