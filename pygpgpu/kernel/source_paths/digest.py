# The content digest that decides whether a generated file still describes its python.
from __future__ import annotations
import hashlib
import sys
from types import CodeType, ModuleType
from typing import List, Optional, Set



#: Where the digest is kept beside a generated source: a sidecar file, never a comment
#: inside the source (see `digest_sidecar`).
DIGEST_SUFFIX: str = ".source-digest"



def _iter_code_consts(consts:tuple)->List[CodeType]:
    """The nested code objects of one code object's constants (`def`, comprehension, ...)."""
    return [value for value in consts if isinstance(value, CodeType)]



def code_digest(code:CodeType)->str:
    """A content digest of a code object, independent of where it was compiled from.

    This replaces the md5 of a module's *text*, and the two properties it has to keep are
    exactly why it is written this way.

    **It must survive freezing.** A frozen application has no `.py` on disk: `co_filename`
    is a bare relative name, and `read()` on it raises `FileNotFoundError`. The bytecode is
    still there -- PyInstaller packs marshalled code objects into the PYZ archive and
    answers `loader.get_code()` -- so a digest taken from the code object works in both
    kinds of process. It is also byte-identical between them, which is what lets a cache
    warmed at build time be found at run time.

    **It must not depend on the path.** `marshal.dumps(code)` looks like the obvious answer
    and is wrong for this: the marshalled payload embeds `co_filename`, so the same source
    compiled in two directories digests differently. `co_filename` is therefore left out of
    the walk, and a code object is identified by what it *does*.

    **`co_code` alone is not enough.** The instruction stream does not carry constants or
    names: `out[i] = 1.5` and `out[i] = 2.5` compile to the same `co_code`, and so do
    `x + helper` and `x + other`. A digest built from `co_code` would answer "unchanged" for
    a body whose generated device code is different, and the stale kernel would keep its
    declarations and its cached binary. The names, the varnames and the constants are
    folded in for that reason, and nested code objects are walked recursively so a helper's
    own body counts.

    Deliberately **not** identical across python versions: the bytecode of one source
    differs between interpreters, and a cached binary from another version should not be
    trusted anyway.
    """
    accumulator:hashlib._Hash = hashlib.md5()
    # `seen` guards a code object that references itself through a constant, which python
    # does not produce today but a hand-built tree could.
    seen:Set[int] = set()

    def walk(current:CodeType)->None:
        if id(current) in seen:
            return
        seen.add(id(current))
        accumulator.update(current.co_code)
        accumulator.update(repr(current.co_names).encode("utf-8"))
        accumulator.update(repr(current.co_varnames).encode("utf-8"))
        accumulator.update(repr(current.co_freevars).encode("utf-8"))
        accumulator.update(repr(current.co_cellvars).encode("utf-8"))
        accumulator.update(repr(current.co_flags).encode("utf-8"))
        for value in current.co_consts:
            if isinstance(value, CodeType):
                walk(value)
            else:
                # `repr` rather than a type-by-type encoder: a constant is a literal value
                # (a number, a string, a tuple of them, a frozenset, None), and repr is
                # stable for all of them within one interpreter.
                accumulator.update(repr(value).encode("utf-8"))

    walk(code)
    return accumulator.hexdigest()



def _module_code(module:ModuleType)->Optional[CodeType]:
    """The module's own code object, through its loader, or None when it cannot be read.

    Two paths, because the two kinds of process expose it differently. A frozen module is
    served by PyInstaller's loader, whose `get_code` answers the marshalled code object from
    the archive. An ordinary module is served by `SourceFileLoader`, which also has
    `get_code`, so one call covers both -- and the `exec`/`-c` cases simply have no loader
    and fall through to None.
    """
    loader:object = getattr(getattr(module, "__spec__", None), "loader", None)
    get_code:object = getattr(loader, "get_code", None)
    if not callable(get_code):
        return None
    try:
        code:object = get_code(module.__name__)
    except Exception:
        # A loader that cannot answer must not turn a cache lookup into a launch failure.
        return None
    return code if isinstance(code, CodeType) else None



def module_digest(module:ModuleType)->str:
    """The digest of one python module's code, or "" when it cannot be read.

    Per **module**, not per function, and that is load-bearing: one generated source file
    holds the device code of every kernel and helper written in that module, and its digest
    sidecar is a single value. A per-function digest would make two kernels of one module
    fight over that sidecar, each invalidating the other's file on every launch -- which is
    exactly what happened when this fell back to per-function for `__main__`.
    """
    code:Optional[CodeType] = _module_code(module)
    if code is None:
        return ""
    return code_digest(code)



def _main_module_digest(code:CodeType)->str:
    """A module-scoped digest for a module no loader serves.

    `__main__` (a script run directly) has no `__spec__`, no `loader.get_code`, and no
    module-level code object reachable from a function, so the module's own bytecode cannot
    be digested. The fallback must still be *module*-scoped, or two kernels of one script
    digest differently and keep invalidating each other's generated file.

    Two answers, in order:

    * **The source file's text**, when it can be read. That is module-scoped and
      content-sensitive, which is what the freshness test wants while developing.
    * **The recorded file name alone**, when it cannot. A frozen application's `__main__` is
      this case: the bundle is immutable for the life of the artifact, so there is nothing
      for a content digest to detect -- and the name still separates two applications whose
      kernels could otherwise share a cache entry.
    """
    text_digest:str = _text_digest(code.co_filename)
    if text_digest:
        return text_digest
    return hashlib.md5(
        _normalize_source_path(code.co_filename).encode("utf-8")).hexdigest()



def object_source_digest(obj:object)->str:
    """The digest of the module an object was defined in, falling back to its own code.

    A device helper defined in another module is filed under *that* module's generated
    source, so its file's freshness has to be decided by that module -- not by the kernel
    that happened to call it.
    """
    module_name:str = getattr(obj, "__module__", "") or ""
    module:Optional[ModuleType] = sys.modules.get(module_name) if module_name else None
    if module is not None:
        digest:str = module_digest(module)
        if digest:
            return digest
    code:object = getattr(obj, "__code__", None)
    if isinstance(code, CodeType):
        # A module with no loader (`__main__`, an `exec`-defined function). The answer stays
        # module-scoped through the recorded file name; see `_main_module_digest`.
        return _main_module_digest(code)
    return ""



def file_source_digest(source_file:str)->str:
    """The digest of the module a python file belongs to, found by its recorded name.

    `code_digest` needs the code object, and a device helper's entry only records the
    `co_filename` of the function it was translated from. So the loaded modules are asked
    which one came from that file, and that module's code is digested.

    Matching is on the recorded name, which is exact while unfrozen and *relative* while
    frozen (`usermod/kern.py`), so the comparison uses the same normalized form on both
    sides: a suffix match against each module's `co_filename` and `__file__`. A module that
    cannot be found falls back to reading the file, which is what an unfrozen process and a
    module outside `sys.modules` both want.
    """
    normalized:str = _normalize_source_path(source_file)
    if normalized:
        for module in list(sys.modules.values()):
            if not isinstance(module, ModuleType):
                continue
            recorded:str = _normalize_source_path(getattr(module, "__file__", "") or "")
            code:Optional[CodeType] = None
            if recorded and _same_file(recorded, normalized):
                digest:str = module_digest(module)
                if digest:
                    return digest
                continue
            code = _module_code(module)
            if code is None:
                continue
            if _same_file(_normalize_source_path(code.co_filename), normalized):
                return code_digest(code)
    return _text_digest(source_file)



def _normalize_source_path(path:str)->str:
    """A comparable spelling of a source path: forward slashes, no leading dot-segments."""
    if not path:
        return ""
    return path.replace("\\", "/").lstrip("./")



def _same_file(left:str, right:str)->bool:
    """Whether two normalized source paths name the same file.

    A frozen module records a relative path and the kernel records the same relative path,
    so equality usually decides. A suffix match covers the case where one side carries a
    directory prefix the other does not (`app/usermod/kern.py` against `usermod/kern.py`),
    which is what a build directory adds.
    """
    if not left or not right:
        return False
    if left == right:
        return True
    return left.endswith("/" + right) or right.endswith("/" + left)



def _text_digest(source_python:str)->str:
    """The md5 of a file's text, or "" when it cannot be read."""
    try:
        with open(source_python, "r", encoding="utf-8") as file:
            return hashlib.md5(file.read().encode("utf-8")).hexdigest()
    except OSError:
        return ""
