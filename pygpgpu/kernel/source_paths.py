"""Where a generated device source lives, and how several of them share one file.

A kernel's device code is a C source file, and one file is one *compilation unit*: a
device function must be declared before its caller, and a kernel that calls a helper can
only be compiled as one translation unit. So the split of the generated code follows the
split of the python code it came from:

* two python functions in one module generate device code in one source file, and a kernel
  and a helper defined beside it stay in that same file -- no include is involved at all;
* two python functions in different modules generate device code in two source files, and
  the caller `#include`s the callee.

The directory mirrors the python file's own place in the tree, relative to the directory
that holds the kernel. A path that would leave that directory (the kernel's module sits
below a helper's) cannot be spelled with `..`, so the parent segment becomes `__`: the
result is still inside the kernel's own cache.

Standard-library and installed-package helpers go to `Lib/` and `Lib/site-packages/`,
mirroring their own python path, because a package's location is not part of the caller's
project and a `/site-packages/torch/...` directory chain inside the caller's cache would
be noise.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import site
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

# The extension a generated source needs, per backend.
EXTENSIONS: Dict[str, str] = {"opencl": "cl", "cuda": "cu", "hip": "hip", "metal": "metal"}
# The cache folder a backend writes into.
CACHE_FOLDERS: Dict[str, str] = {"opencl": "__clcache__", "cuda": "__cucache__",
                                 "hip": "__hipcache__", "metal": "__metalcache__"}

# The directory name for a python file from the standard library, and for one from an
# installed package: both mirror their own path below these roots.
LIB_FOLDER: str = "Lib"
SITE_PACKAGES_FOLDER: str = os.path.join("Lib", "site-packages")

# One generated file belongs to one python module, and several threads (and several
# processes) can compile different kernels of that module at the same time. The write is a
# read-modify-write -- read the file, merge the definitions this call rendered into it, and
# write it back -- so without a lock the later writer drops what the earlier one added, and
# the kernel that lost its definition fails to load with a bare `KeyError` naming the entry
# point it cannot find. Two locks are needed and neither replaces the other:
#
# * a process-wide lock per file (the table below), because a `threading.Lock` is what makes
#   this atomic between threads, and threads in one process share every file;
# * an OS-level lock file beside the target (see `_file_lock`), because separate processes
#   share nothing in memory -- the `__clcache__` directory is on disk and every process
#   writing the same module writes the same path.
#
# The process lock is re-entrant: a kernel that reaches a helper compiles both, and the
# helper's own write may land inside the caller's critical section.
_WRITE_LOCKS: Dict[str, threading.RLock] = {}
_WRITE_LOCKS_GUARD = threading.Lock()
# The files whose lock *this thread* already holds, so `_file_lock` can re-enter rather
# than wait on itself. Thread-local by construction: a second thread must wait, and only
# the owner re-enters.
_file_lock_state = threading.local()
# How long a write waits for another process, and how long one attempt of the lock file
# lasts. A compile is seconds at worst; ten minutes is "hang detection", not a budget.
_FILE_LOCK_TIMEOUT: float = 600.0
_FILE_LOCK_POLL: float = 0.05
# How many times `_ensure_directory` retries a create another writer is racing.
_MKDIR_ATTEMPTS: int = 8
# How long one generated-source write keeps retrying while another process holds the
# destination. A scanner's hold is milliseconds and an indexer's is a second at most, so a
# couple of seconds is generous; the point is that a *transient* hold never fails a launch,
# while a genuinely stuck one still reports itself instead of hanging forever.
_SOURCE_WRITE_TIMEOUT: float = 5.0
_SOURCE_WRITE_POLL: float = 0.05



def _standard_library_directory() -> str:
    """The directory python keeps its own modules in, as a normalized absolute path."""
    candidate: str = os.path.dirname(os.path.abspath(os.__file__))
    return os.path.normcase(candidate) if os.path.isdir(candidate) else ""


def _site_packages_directories() -> List[str]:
    """Every installed-package directory, as normalized absolute paths.

    `site.getsitepackages()` also lists the installation *root* on Windows
    (`D:\\Python\\314`), which is where the standard library lives rather than packages, so
    only a directory that actually holds the `site-packages` name counts. The deepest one
    wins, because the root contains every installation's own folder.
    """
    directories: List[str] = []
    entries: List[str] = []
    try:
        entries.extend(site.getsitepackages())
    except AttributeError:                      # a virtualenv without that helper
        pass
    try:
        entries.append(site.getusersitepackages())
    except AttributeError:
        pass
    for candidate in entries:
        if not candidate or not os.path.isdir(candidate):
            continue
        normalized: str = os.path.normcase(os.path.abspath(candidate))
        # A package directory is named `site-packages` (or `dist-packages` on Debian).
        if not (os.path.basename(normalized) in ("site-packages", "dist-packages")
                or normalized.endswith(os.sep + "site-packages")):
            continue
        directories.append(normalized)
    return directories


def _is_within(path: str, root: str) -> bool:
    """Whether `path` is inside `root` (both already normalized)."""
    if not root:
        return False
    return path.startswith(root + os.sep) or path == root


def _library_relative(source_directory: str) -> Optional[Tuple[str, str]]:
    """Where a standard-library or installed-package module's code belongs.

    Returns (root folder, path below it) or None when the module is the caller's own code.
    """
    normal_source = os.path.normcase(os.path.abspath(source_directory))
    # The standard library is checked FIRST: a package directory sits *inside* it on
    # Windows (`D:\\Python\\314\\Lib\\site-packages`), so testing the other way round
    # classified the standard library itself as an installed package.
    standard = _standard_library_directory()
    if _is_within(normal_source, standard):
        below = normal_source[len(standard) + len(os.sep):]
        if below and _is_within(normal_source, os.path.join(
                standard, "site-packages")):
            packages_root = os.path.join(standard, "site-packages")
            return SITE_PACKAGES_FOLDER, normal_source[
                len(packages_root) + len(os.sep):]
        return LIB_FOLDER, below
    for root in _site_packages_directories():
        if _is_within(normal_source, root):
            below = normal_source[len(root) + len(os.sep):] if normal_source != root else ""
            return SITE_PACKAGES_FOLDER, below
    return None


def _sanitize(segment: str) -> str:
    """A path segment that is a legal folder name on every filesystem this targets."""
    cleaned = "".join(character if (character.isalnum() or character in "._-")
                      else "_" for character in segment)
    return cleaned or "__"


def _relative_segments(path: str) -> List[str]:
    """The path's own segments, with a leading parent segment written as `__`.

    A path that names the directory itself (`"."`) has no segments: the caller decides how
    to name that case, and returning `["."]` would put a dot in every folder name.
    """
    segments: List[str] = []
    for part in path.replace("\\", "/").split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            # A parent cannot be spelled: the cache may not leave the kernel's own folder,
            # so the segment is the marker the reader can still see it was a parent.
            segments.append("__")
        else:
            segments.append(_sanitize(part))
    return segments


def package_source_files() -> List[str]:
    """Every python file of this package, as normalized absolute paths.

    A structure declaration is routed to the module that defines its class, but a class can
    live here instead of in the caller's project: a structure synthesized from a plain
    python object is built by this package, and two classes can share one layout, so the
    class registered first may be one of ours. Writing a caller's declaration into a path
    inside this package is never right, so this list is what a caller's own module is
    checked against.
    """
    package_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files: List[str] = []
    for directory, _subdirectories, names in os.walk(package_root):
        for name in names:
            if name.endswith(".py"):
                files.append(os.path.normcase(
                    os.path.abspath(os.path.join(directory, name))))
    return files


def source_directory_of(kernel_file: str) -> str:
    """The directory of the python file a kernel is defined in."""
    return os.path.dirname(os.path.abspath(kernel_file)) if kernel_file else os.getcwd()


def module_directory(kernel_directory: str, source_file: str) -> str:
    """The cache subdirectory one python module's device code is written to.

    The kernel's own directory needs no wrapper: `__xxcache__` already sits beside the
    python files, so a module next to the kernel writes straight into `pykernels/`, and a
    module reached from a subpackage gets the subpath that leads to it.
    """
    absolute = os.path.abspath(source_file) if source_file else ""
    directory = os.path.dirname(absolute)
    library = _library_relative(directory)
    if library is not None:
        root, below = library
        return os.path.join(root, *_relative_segments(below)) if below else root

    segments = _relative_segments(os.path.relpath(directory, kernel_directory))
    if not segments:
        # The kernel's own directory: nothing to nest under.
        return ""
    return os.path.join(*segments)


def source_path(kernel_file: str, source_file: str, backend: str) -> str:
    """The generated source file one python module's device code is written to.

    The file keeps the python module's own name: `scene.py` generates `scene.cu`, so a
    reader can find the code behind a name without looking anything up.
    """
    extension: str = EXTENSIONS[backend]
    name: str = os.path.basename(source_file) if source_file else "kernel.py"
    stem: str = os.path.splitext(name)[0] or "kernel"
    directory: str = module_directory(source_directory_of(kernel_file), source_file)
    return os.path.join(cache_root(kernel_file, backend), directory,
                        f"{_sanitize(stem)}.{extension}")


def cache_root(kernel_file: str, backend: str) -> str:
    """The `pykernels` folder a kernel's generated sources live in."""
    return os.path.join(source_directory_of(kernel_file), CACHE_FOLDERS[backend],
                        "pykernels")


def guard_macro(path: str) -> str:
    """The include guard a generated file's own path gives it.

    Two generated files can declare the same structure (a `Ray` in two modules has one
    layout, therefore one C name), so a source that reaches both would define the type
    twice -- which C++ rejects. A guard per file fixes exactly that: the second inclusion
    is skipped whole, and its declarations come from the first.
    """
    digest: str = hashlib.md5(
        os.path.normcase(os.path.abspath(path)).encode("utf-8")).hexdigest()[:12]
    # `__<file>_<hash>__`: the file's own name is the readable half, the hash keeps two
    # same-named modules (both `scene.cu`, say) apart, and the trailing underscores mark
    # the end the way a leading pair marks the start.
    stem: str = os.path.splitext(os.path.basename(path))[0] or "source"
    return f"__{_sanitize(stem)}_{digest}__"


def guard_source(path: str, source: str) -> str:
    """Wrap one generated source in its include guard."""
    macro: str = guard_macro(path)
    return f"#ifndef {macro}\n#define {macro}\n\n{source.rstrip()}\n\n#endif\n"


def include_line(referrer_path: str, callee_path: str) -> str:
    """The `#include` that makes one generated file reach another.

    A quoted include resolves relative to the file that writes it, so the callee is spelled
    relative to the referrer; both live inside the same cache root, so the path never
    climbs out of it.
    """
    referrer_directory: str = os.path.dirname(os.path.abspath(referrer_path))
    target: str = os.path.relpath(os.path.abspath(callee_path), referrer_directory)
    return '#include "%s"' % target.replace("\\", "/")


def _definition_identity(code: str) -> Optional[str]:
    """A stable identity for one chunk of generated device code.

    A chunk is either a whole C function (`<qualifiers> <type> <name>(<params>)` followed by
    its body) or a structure typedef block. The function's own header line identifies it
    (the same helper specialized twice differs in the parameters, so those stay distinct);
    a structure is identified by the type name it declares, because its member list is part
    of the same declaration and must not be emitted twice.
    """
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("struct "):
            return stripped
        if stripped.startswith("typedef struct"):
            # OpenCL spells a structure as `typedef struct { ... } Name;`
            continue
        if "(" in stripped and ";" not in stripped:
            return stripped
    # An OpenCL structure block ends in its own name; that is its identity.
    for line in reversed(code.splitlines()):
        stripped = line.strip()
        if stripped.startswith("}") and stripped.endswith(";"):
            return stripped
    return None


def merge_source(existing: str, addition: str) -> str:
    """`existing` with every definition of `addition` it does not already have.

    Python functions in one module have distinct names, so a definition cannot repeat; this
    only guards against a source being regenerated with pieces the file already holds
    (a kernel that was compiled, its file kept, and the same kernel compiled again).
    """
    if not existing.strip():
        return addition
    signatures: set = set()
    for chunk in existing.split("\n\n"):
        signature = _definition_identity(chunk)
        if signature is not None:
            signatures.add(signature)
    pieces: List[str] = []
    for chunk in addition.split("\n\n"):
        signature = _definition_identity(chunk)
        if signature is not None and signature in signatures:
            continue
        pieces.append(chunk)
        if signature is not None:
            signatures.add(signature)
    if not pieces:
        return existing
    return existing.rstrip("\n") + "\n\n" + "\n\n".join(piece.rstrip("\n")
                                                          for piece in pieces) + "\n"


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
    `#include` lines and preprocessor defines. Those belong to the file as a whole and are
    emitted once, whichever kernel happens to be written second. Everything else is a
    function definition.
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
                                or first.startswith("#"))
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


QUAT_DEFINE_PREFIX: str = "#define QUAT_"


def _is_file_guard_line(line: str, macro: str = "") -> bool:
    """Whether a line belongs to a generated file's own include guard.

    The guard names the file (`__scene_<hash>__`), so it is matched by that macro when the
    caller knows it; the pattern covers the reading path, which strips a file's guard without
    knowing which file it came from.
    """
    stripped: str = line.strip()
    if macro:
        return stripped in (f"#ifndef {macro}", f"#define {macro}")
    return (stripped.startswith("#ifndef __") and stripped.endswith("__")
            and "_source_" not in stripped)


def _is_include_only(piece: str) -> bool:
    """Whether a piece is nothing but `#include` lines."""
    lines: List[str] = [line for line in piece.splitlines() if line.strip()]
    return bool(lines) and all(line.lstrip().startswith("#include") for line in lines)


def _strip_file_guard(source: str, path: str) -> str:
    """The file's body, without the guard the writer put around it.

    The guard's `#endif` is the file's last line, so it is removed first and the `#ifndef`
    line only when it names this file's own guard -- a structure guard is a different
    macro and has to stay.
    """
    macro: str = guard_macro(path) if path else ""
    lines: List[str] = source.splitlines()
    if macro:
        lines = [line for line in lines
                 if line.strip() not in (f"#ifndef {macro}", f"#define {macro}")]
    while lines and not lines[0].strip():
        lines.pop(0)
    # The closing `#endif` of the file guard is the last non-empty line, and only when the
    # guard was stripped above.
    if macro:
        for index in range(len(lines) - 1, -1, -1):
            if not lines[index].strip():
                continue
            if lines[index].strip() == "#endif":
                lines.pop(index)
            break
    return "\n".join(lines)


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
    for piece in header:
        lines: List[str] = piece.splitlines()
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
    # The declarations come before the includes: a file this source includes can name a
    # structure declared here (a helper in another module takes a `Ray` by pointer), and
    # the compiler reads the included text after these lines.
    canonical.extend(declarations)
    canonical.extend(includes)
    return canonical


def python_source_digest(source_python:str)->str:
    """A digest of the python text that produced a generated file.

    The freshness test used to be mtime alone, which assumes a file's timestamp moves
    forward when its text changes. Git breaks that assumption: `git checkout` and
    `git stash pop` restore a file *and* set its mtime back, so a generated file written
    from newer text kept looking "newer than the source" and was reused -- leaving the old
    declaration of a kernel beside the new one, and the compiler reported "more than one
    instance of overloaded function". Reading the text is what makes the test about
    content rather than about a clock.
    """
    try:
        with open(source_python, "r", encoding="utf-8") as file:
            return hashlib.md5(file.read().encode("utf-8")).hexdigest()
    except OSError:
        return ""


def _path_lock(path:str)->threading.RLock:
    """The process-wide lock for one generated file, created on first ask."""
    key: str = os.path.normcase(os.path.abspath(path))
    with _WRITE_LOCKS_GUARD:
        lock = _WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _WRITE_LOCKS[key] = lock
    return lock


def _ensure_directory(directory:str)->None:
    """Make `directory` exist; the tolerant implementation lives in `utils`."""
    from ..utils import _ensure_directory as ensure

    ensure(directory)


def _write_source_file(path:str, source:str)->None:
    """Replace one generated file's contents, surviving a transient external hold.

    A plain `open(path, "w")` fails with `PermissionError` when *another process* holds the
    destination with a non-shared handle. Measured on Windows: a same-process read handle
    does **not** block a plain write, but it does block `os.replace`; an exclusive handle
    held by another process blocks both. Since the generated source is rewritten hundreds of
    times while a module's kernels compile, a real-time scanner or an indexer taking the file
    for a moment lands inside one of those windows and the whole launch fails -- which is
    the intermittent error this code exists to remove.

    So the write is tolerant rather than optimistic: it tries `os.replace` over a temp file
    (which is what makes a reader never see a half-written file), and on `PermissionError`
    it retries for a bounded time -- alternating the atomic form with a plain overwrite,
    because the two fail for *different* kinds of holder. Only when the hold outlives the
    whole budget does the error surface, and then it names the file and the holder hint.
    """
    temp_name:str = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    deadline:float = time.monotonic() + _SOURCE_WRITE_TIMEOUT
    payload:bytes = source.encode("utf-8")
    last_error:Optional[PermissionError] = None
    while True:
        try:
            with open(temp_name, "wb") as temp_file:
                temp_file.write(payload)
            os.replace(temp_name, path)
            return
        except PermissionError as error:
            last_error = error
        try:
            # A plain overwrite is refused by a *different* set of holders than `os.replace`
            # is, so trying both each pass covers whichever one is in the way.
            with open(path, "wb") as file:
                file.write(payload)
            return
        except PermissionError as error:
            last_error = error
        finally:
            with contextlib.suppress(OSError):
                if os.path.isfile(temp_name):
                    os.unlink(temp_name)
        if time.monotonic() >= deadline:
            raise PermissionError(
                f"cannot write {path}: another process held it for more than"
                f" {_SOURCE_WRITE_TIMEOUT:.0f}s (a real-time scanner or an indexer holding a"
                f" generated file is the usual cause). Last error: {last_error}") from None
        time.sleep(_SOURCE_WRITE_POLL)


def _file_lock_path(path:str)->str:
    """Where the cross-process lock for one generated file lives.

    Beside the file it guards, with a `.lock` suffix: a lock file that outlives a crash
    carries no state to get wrong, because ownership is the OS handle itself -- a killed
    process releases it, so a stale file is an empty file nobody holds, not a deadlock.
    """
    return path + ".lock"


@contextlib.contextmanager
def _file_lock(path:str)->Iterator[None]:
    """Hold the cross-process lock for one generated file.

    Two processes compiling different kernels of the same python module write the same
    generated file, and the write is a read-modify-write: without this, the process that
    reads first and writes last drops the definitions the other one added. The lock is an
    OS-level exclusive create (`O_CREAT | O_EXCL`) on a `.lock` file beside the target,
    which is atomic on every platform this package targets and needs no library.

    The wait is bounded rather than infinite, so a genuinely stuck holder reports itself
    instead of hanging the caller forever. A lock file left behind by a killed process is
    empty and unheld, so an attempt that finds one simply retries and then takes it: the
    ownership is the open handle, not the file's existence.

    A thread that already holds this file's lock re-enters it without taking the lock file
    again. An exclusive create has no notion of "already mine", so a nested acquisition --
    a helper's write inside its caller's critical section, or the write-and-compile pair
    around a kernel that reaches a helper -- would wait on a lock file this very thread is
    holding, which is a deadlock rather than contention. The held set is per thread, so no
    other thread is mistaken for the owner.
    """
    key: str = os.path.normcase(os.path.abspath(path))
    held: set = getattr(_file_lock_state, "held", None) or set()
    if key in held:
        # This thread already owns the file's lock; taking it again would wait on itself.
        yield
        return

    lock_path: str = _file_lock_path(path)
    _ensure_directory(os.path.dirname(lock_path) or ".")
    deadline: float = time.monotonic() + _FILE_LOCK_TIMEOUT
    handle: Optional[int] = None
    while True:
        try:
            # `O_EXCL` makes "create if absent" one atomic step, which is what "I own this
            # file now" means here. Every later writer sees EEXIST and waits.
            handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out after {_FILE_LOCK_TIMEOUT:.0f}s waiting for another process "
                    f"to finish writing {path}; if no other process is compiling, delete "
                    f"{lock_path} and retry") from None
            time.sleep(_FILE_LOCK_POLL)
    held.add(key)
    _file_lock_state.held = held
    try:
        yield
    finally:
        held.discard(key)
        if handle is not None:
            os.close(handle)
        with contextlib.suppress(OSError):
            os.unlink(lock_path)


@contextlib.contextmanager
def write_lock(path:str)->Iterator[None]:
    """Both the thread lock and the process lock for one generated file.

    The order is fixed -- thread lock first, then the lock file -- so two writes cannot
    deadlock against each other, and a thread that already holds its own lock (a helper's
    write nested inside its caller's) simply re-enters it.
    """
    with _path_lock(path):
        with _file_lock(path):
            yield


def append_source(path: str, header: List[str], definitions: List[str],
                  source_python: str, source_digest: str = "") -> str:
    """Write definitions into a generated file, keeping what the file already holds.

    One python module owns one generated file, and it is compiled once per kernel, so each
    call contributes the definitions it rendered and must not drop the ones an earlier call
    wrote. Python functions in one module have distinct names, so the union cannot collide;
    a definition the file already has is skipped, which is what makes a recompiled kernel
    idempotent.

    The whole read-merge-write is one critical section: two threads or two processes
    compiling different kernels of this module reach this function with the same path, and
    an unserialized writer would base its merge on a file that no longer holds what the
    other one added. See `write_lock`.
    """
    with write_lock(path):
        return _append_source_locked(path, header, definitions, source_python, source_digest)


def _append_source_locked(path: str, header: List[str], definitions: List[str],
                          source_python: str, source_digest: str) -> str:
    """`append_source`'s body, with the file's locks already held."""
    existing: Optional[str] = None
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                existing = file.read()
        except OSError:
            existing = None

    if existing is not None:
        # The digest sidecar is the authority when it exists: it describes the python
        # text that wrote this file, so a matching digest means nothing changed even
        # when the timestamps say the source is newer (`git checkout` sets the mtime
        # back, and two files written inside one clock tick compare equal). The clock
        # is only consulted when no digest was ever recorded -- a file from before the
        # sidecar existed, or one written without a digest -- and even then it only
        # decides the ambiguous direction: an equal timestamp is not evidence that the
        # source changed, so the file is kept.
        stored: str = _stored_digest(path)
        if source_digest and stored:
            target_is_newer = stored == source_digest
        else:
            try:
                target_is_newer = (
                    os.path.isfile(source_python)
                    and os.path.getmtime(path) >= os.path.getmtime(source_python))
            except OSError:
                target_is_newer = False
        if not target_is_newer:
            # The file describes python text that has since changed: start it over.
            existing = None

    known: set = set()
    kept_definitions: List[str] = []
    kept_header: List[str] = []
    if existing is not None:
        kept_header, kept_definitions = _split_source(existing, path)
        for piece in kept_definitions:
            identity = _definition_identity(piece)
            if identity is not None:
                known.add(identity)

    added: List[str] = []
    for piece in definitions:
        identity = _definition_identity(piece)
        if identity is not None and identity in known:
            continue
        added.append(piece.strip("\n"))
        if identity is not None:
            known.add(identity)
    # Nothing is skipped here even when every definition is already known: the header may
    # still gain a quaternion width the earlier run did not need.

    merged_header: List[str] = (list(kept_header) if existing is not None else [])
    for piece in header:
        cleaned = piece.strip("\n")
        if cleaned:
            merged_header.append(cleaned)
    merged_header = _canonical_header(merged_header)
    source: str = assemble_source(merged_header, kept_definitions + added, path)
    _ensure_directory(os.path.dirname(path))
    # An unchanged file is not rewritten. Every rewrite opens a window in which a scanner
    # or an indexer can hold the destination exclusively, and the module is re-rendered on
    # every kernel call -- 265 rewrites of one file in a single test module, most of them
    # byte-identical. Skipping the no-op write removes those windows entirely, and the
    # digest is still refreshed because it is a separate, tolerant write.
    if existing != source:
        _write_source_file(path, source)
    _write_digest(path, source_digest)
    return path


DIGEST_SUFFIX: str = ".source-digest"


def digest_sidecar(path:str)->str:
    """Where a generated file's source digest is kept.

    A sidecar rather than a comment inside the source: the generated text is a
    carefully-assembled translation unit whose blank lines are significant (`_split_source`
    splits it on a blank line), so threading one more line through it risked changing how a
    later write re-reads the file. A `.source-digest` file next to the source cannot affect
    what the compiler sees.
    """
    return path + DIGEST_SUFFIX


def _write_digest(path:str, digest:str)->None:
    if not digest:
        return
    try:
        with open(digest_sidecar(path), "w", encoding="utf-8", newline="\n") as file:
            file.write(digest + "\n")
    except OSError:
        pass


def _stored_digest(path:str, source:str="")->str:
    """The digest recorded beside a generated file, or "" when there is none."""
    try:
        with open(digest_sidecar(path), "r", encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return ""


def write_source(path: str, source: str, source_python: str,
                 keep_definitions: bool = True) -> str:
    """Write one module's generated source, honouring the append/overwrite rule.

    A target file **newer** than the python file it came from is kept and the new
    definitions are appended to it: the file was written after that python text, so it
    holds work this run should not throw away. An **older** target is replaced outright, it
    describes python text that has since changed.

    `keep_definitions` makes an overwrite keep the definitions the file already holds, which
    is what one python module needs: two kernels defined beside each other live in one
    source file, and each is compiled on its own call, so the second call must not erase the
    first kernel. Python functions in one module have distinct names, so adding them cannot
    collide.

    Like `append_source`, this is a read-modify-write over a file several threads and
    processes may be writing, so it runs under the same locks.
    """
    with write_lock(path):
        return _write_source_locked(path, source, source_python, keep_definitions)


def _write_source_locked(path: str, source: str, source_python: str,
                         keep_definitions: bool) -> str:
    """`write_source`'s body, with the file's locks already held."""
    _ensure_directory(os.path.dirname(path))
    existing: Optional[str] = None
    if os.path.isfile(path):
        existing = None
        try:
            with open(path, "r", encoding="utf-8") as file:
                existing = file.read()
        except OSError:
            existing = None

    if existing is not None and os.path.isfile(source_python):
        try:
            target_is_newer: bool = (os.path.getmtime(path)
                                     > os.path.getmtime(source_python))
        except OSError:
            target_is_newer = False
        if target_is_newer:
            merged: str = merge_source(existing, source)
            if merged != existing:
                _write_source_file(path, merged)
            return path

    if existing is not None and keep_definitions:
        # The file is this module's whole device code, and this call renders only one
        # kernel plus the helpers it reaches. Keep what the file already holds as well, so
        # a module with two kernels keeps both.
        combined: str = merge_source(existing, source)
        if combined != existing:
            _write_source_file(path, combined)
        return path

    if existing != source or existing is None:
        _write_source_file(path, source)
    return path
