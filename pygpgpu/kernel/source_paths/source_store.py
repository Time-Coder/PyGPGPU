# A module's python source, kept beside the device code it generated.
"""The kernel bodies a frozen application can no longer read for itself.

Every launch starts by reading a kernel's python source and parsing it: the translator needs
the AST to emit C, and the launch plan needs it for the host regions, the baked shared-buffer
shapes and the subscript checks. `inspect.getsource` is how that text is obtained, and it
fails wherever no `.py` sits on disk -- which is exactly the frozen case: PyInstaller keeps
the *bytecode* (marshalled code objects inside the PYZ archive) and drops the source.

So the text is remembered where the generated device code already lives, in the same
`__xxcache__` tree that holds the `.cu`/`.cl` and its digest. The store is written the first
time a module is translated, which costs one file read per module -- the same read
`inspect.getsource` has just done -- and it needs no extra step from a caller. A tree that is
shipped inside a packaged application then carries the bodies along with the binaries, so a
launch in the packaged process can still translate.

This is deliberately a *source* store and not a serialized AST: a pickle of a parse tree is
tied to the CPython version that wrote it, while text is re-parsed by whatever interpreter
reads it, through the same normalization every other parse goes through
(`source_text._parse_kernel_source`).
"""
from __future__ import annotations
import ast
import os
import sys
from typing import List, Optional, Tuple

from .layout import CACHE_FOLDERS, cache_root



#: The folder inside one backend's cache that holds remembered sources.
SOURCE_FOLDER: str = "sources"


#: The suffix a remembered module is stored under.
#:
#: Deliberately **not** `.py`: a file called `x.py` inside a cache tree is indistinguishable
#: from a real source module to anything that walks the tree looking for python -- pytest
#: collects it (it sits under `tests/`, so it is picked up as a second module of the same
#: name and the run aborts with "import file mismatch"), a linter type-checks it, and a
#: packager may ship it as code. The store is a *record*, so it gets a record's suffix and no
#: tool mistakes it for a module.
SOURCE_SUFFIX: str = ".pysrc"



def _safe_module_name(module_name:str)->str:
    """A module name as a relative path: `pkg.mod` -> `pkg/mod.pysrc`.

    A dotted module name maps onto the directory shape its file would have, so one reader can
    look a module up from either the name or the path without a second index.
    """
    cleaned:str = module_name.strip(".")
    if not cleaned:
        return ""
    return "/".join(part for part in cleaned.split(".") if part) + SOURCE_SUFFIX



def _module_key(module_name:str, kernel_file:str)->str:
    """The name a module's source is stored under.

    The module name when there is one, because that is what a frozen process still knows (a
    frozen module keeps `__name__`; its `co_filename` is a bare relative path). A kernel from
    an unnamed module (`__main__` of a script) falls back to its file name.
    """
    key:str = _safe_module_name(module_name)
    if key and key != SOURCE_SUFFIX:
        return key
    return _safe_module_name(os.path.splitext(os.path.basename(kernel_file))[0])



def _tree_root(kernel_file:str)->str:
    """The directory that holds every backend's cache folder.

    `cache_root` answers `<tree>/__xxcache__/pykernels`, so the tree is two levels up.
    """
    return os.path.dirname(os.path.dirname(cache_root(kernel_file, "cuda")))



def _candidate_roots(kernel_file:str)->List[str]:
    """Every cache *tree* that could hold this module's source.

    Two roots, because a frozen process and an ordinary one resolve `co_filename` differently:

    * the tree `cache_root` derives from `co_filename` -- the ordinary case, where the python
      file and its cache sit together;
    * the *bundle* root (`sys._MEIPASS`), which a frozen application extracts its data files
      into. A frozen module's `co_filename` is only a bare relative name, so the derived tree
      would point at the process's working directory instead, where nothing was shipped.

    The bundle root is appended rather than preferred: while developing, both name the same
    tree and the first answer wins without an extra stat.
    """
    roots:List[str] = [_tree_root(kernel_file)]
    bundle_root:str = getattr(sys, "_MEIPASS", "") or ""
    if bundle_root and os.path.normcase(os.path.abspath(bundle_root)) != os.path.normcase(
            os.path.abspath(roots[0])):
        roots.append(bundle_root)
    return roots



def _store_path(kernel_file:str, backend:str, key:str)->str:
    """Where one module's source lives for one backend."""
    return os.path.join(cache_root(kernel_file, backend), SOURCE_FOLDER,
                        *key.split("/"))



def store_module_source(kernel_file:str, backend:str, module_name:str,
                        text:str)->Optional[str]:
    """Remember one module's source, if it is not already there.

    Returns the path written, or None when the text was empty or nothing had to change. A
    caller that cannot read the source simply does not call this; there is nothing to store.
    """
    if not text:
        return None
    key:str = _module_key(module_name, kernel_file)
    if not key:
        return None
    path:str = _store_path(kernel_file, backend, key)
    if _read_text(path) == text:
        return None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as target:
            target.write(text)
    except OSError:
        # A read-only cache is not a failure of the launch: the store is an accelerant for a
        # *later* process, never a requirement of this one.
        return None
    return path



def load_module_source(kernel_file:str, module_name:str)->Optional[str]:
    """One module's remembered source, from whichever cache holds it.

    Both roots are tried, each across every backend: the text is the python module's own,
    identical whichever backend wrote it, and a process that ships only one backend's cache
    folder should still find it.
    """
    key:str = _module_key(module_name, kernel_file)
    if not key:
        return None
    for root in _candidate_roots(kernel_file):
        for folder in CACHE_FOLDERS.values():
            path:str = os.path.join(root, folder, "pykernels", SOURCE_FOLDER,
                                    *key.split("/"))
            text:Optional[str] = _read_text(path)
            if text is not None:
                return text
    return None



def _read_text(path:str)->Optional[str]:
    """One file's text, decoded the way python itself decodes a source file.

    `utf-8-sig` rather than `utf-8`: a module saved with a byte-order mark is legal python
    and `ast.parse` rejects the BOM character, so the mark is stripped on the way in and
    never reaches the parser.
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as source:
            return source.read()
    except OSError:
        return None



def function_source(module_text:str, qualname:str,
                    first_line:int=0)->Optional[Tuple[List[str], int]]:
    """One function's lines out of a module's text, as `inspect.getsourcelines` answers.

    The qualname narrows the search to the class or function a nested definition sits in, so
    two methods of one name on two classes are not confused. `first_line` disambiguates the
    remaining case -- the same qualname defined twice in one module (a redefinition, or two
    branches that each define it) -- by preferring the candidate nearest the line the caller
    already knows.

    Returns `(lines, start_line)` with the decorators included, which is what
    `inspect.getsourcelines` gives and therefore what `_source_first_line` expects.
    """
    try:
        tree:ast.Module = ast.parse(module_text.lstrip("\ufeff"))
    except SyntaxError:
        return None
    parts:List[str] = [part for part in qualname.split(".") if part and part != "<locals>"]
    if not parts:
        return None
    candidates:List[ast.AST] = []
    _collect_definitions(tree.body, parts, candidates)
    if not candidates:
        return None
    chosen:ast.AST = candidates[0]
    if first_line and len(candidates) > 1:
        chosen = min(candidates, key=lambda node: abs(
            (getattr(node, "lineno", 1) or 1) - first_line))
    start:int = _start_line(chosen, module_text)
    end:int = getattr(chosen, "end_lineno", None) or _last_line(chosen)
    lines:List[str] = module_text.splitlines(keepends=True)
    return lines[start - 1:end], start



def _start_line(node:ast.AST, module_text:str)->int:
    """The first line of a definition, including its decorators.

    `lineno` of a decorated definition points at the `def`, while `inspect.getsourcelines`
    starts at the first decorator. `_trimmed_source` drops the decorators anyway, so the
    difference only matters for the reported line number -- which is what a traceback shows.
    """
    decorators:List[ast.AST] = list(getattr(node, "decorator_list", []) or [])
    if decorators:
        return min(getattr(item, "lineno", node.lineno) for item in decorators)
    return int(getattr(node, "lineno", 1) or 1)



def _last_line(node:ast.AST)->int:
    """The last line a definition spans, for an interpreter that has no `end_lineno`."""
    last:int = int(getattr(node, "lineno", 1) or 1)
    for child in ast.walk(node):
        candidate:int = getattr(child, "end_lineno", None) or getattr(child, "lineno", 0) or 0
        if candidate > last:
            last = candidate
    return last



def _collect_definitions(body:List[ast.stmt], parts:List[str],
                         found:List[ast.AST])->None:
    """Walk the qualname's key parts, gathering every definition it ends at."""
    name:str = parts[0]
    rest:List[str] = parts[1:]
    for statement in body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if statement.name != name:
                continue
            if rest:
                _collect_definitions(statement.body, rest, found)
            else:
                found.append(statement)
        elif isinstance(statement, ast.If):
            # A definition chosen by a branch (`if DEBUG: def k(...)`) is still reachable.
            _collect_definitions(statement.body, parts, found)
            _collect_definitions(statement.orelse, parts, found)
