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

import hashlib
import os
import site
from typing import Dict, List, Optional, Tuple

# The extension a generated source needs, per backend.
EXTENSIONS: Dict[str, str] = {"opencl": "cl", "cuda": "cu", "hip": "hip"}
# The cache folder a backend writes into.
CACHE_FOLDERS: Dict[str, str] = {"opencl": "__clcache__", "cuda": "__cucache__",
                                 "hip": "__hipcache__"}

# The directory name for a python file from the standard library, and for one from an
# installed package: both mirror their own path below these roots.
LIB_FOLDER: str = "Lib"
SITE_PACKAGES_FOLDER: str = os.path.join("Lib", "site-packages")


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


def library_source_files() -> List[str]:
    """Every python file of this package, as normalized absolute paths.

    A structure synthesized from a plain python object is *built* by this package
    (`objects.py` defines the ctypes class), so the class it records names a library file
    rather than the caller's module. Such a structure belongs where it is used, and this
    list is what tells a caller's own module apart from a file of this package.
    """
    package_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files: List[str] = []
    for directory, _subdirectories, names in os.walk(package_root):
        for name in names:
            if not name.endswith(".py"):
                continue
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
        os.path.normcase(os.path.abspath(path)).encode("utf-8")).hexdigest()
    return f"__PYGPGPU_SOURCE_{digest}"


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
    blocks: List[str] = [piece.strip("\n") for piece in header if piece.strip()]
    leading: List[str] = []
    rest: List[str] = []
    for piece in blocks:
        lines: List[str] = piece.splitlines()
        directives: List[str] = [line for line in lines if line.lstrip().startswith("#")]
        remainder: List[str] = [line for line in lines if line not in directives]
        if directives and not remainder:
            # A pure directive block (`#define` runs, `#include` lines): it leads.
            leading.append(piece)
        else:
            # A block that mixes a directive with declarations (a guarded structure) stays
            # whole: its guard has to wrap its own declaration, not sit beside it.
            rest.append(piece)
    parts: List[str] = leading + rest
    parts.extend(piece.strip("\n") for piece in definitions if piece.strip())
    body: str = "\n\n".join(parts) + "\n"
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
                                if not _is_file_guard_line(line)]
        block: str = "\n".join(remainder).strip("\n")
        if block:
            header.append(block)
    return header, definitions


QUAT_DEFINE_PREFIX: str = "#define QUAT_"


def _is_file_guard_line(line: str) -> bool:
    """Whether a line belongs to a generated file's own include guard."""
    stripped: str = line.strip()
    return (stripped.startswith("#ifndef __PYGPGPU_SOURCE_")
            or stripped.startswith("#define __PYGPGPU_SOURCE_"))


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
            # A block that mixes a directive with a declaration -- a structure wrapped in
            # its own `#ifndef` -- is kept whole: pulling the guard out would leave the
            # declaration bare and the guard guarding nothing.
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
    canonical.extend(includes)
    canonical.extend(declarations)
    return canonical


def append_source(path: str, header: List[str], definitions: List[str],
                  source_python: str) -> str:
    """Write definitions into a generated file, keeping what the file already holds.

    One python module owns one generated file, and it is compiled once per kernel, so each
    call contributes the definitions it rendered and must not drop the ones an earlier call
    wrote. Python functions in one module have distinct names, so the union cannot collide;
    a definition the file already has is skipped, which is what makes a recompiled kernel
    idempotent.
    """
    existing: Optional[str] = None
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                existing = file.read()
        except OSError:
            existing = None

    if existing is not None:
        try:
            target_is_newer: bool = (
                os.path.isfile(source_python)
                and os.path.getmtime(path) > os.path.getmtime(source_python))
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(source)
    return path


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
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
                with open(path, "w", encoding="utf-8", newline="\n") as file:
                    file.write(merged)
            return path

    if existing is not None and keep_definitions:
        # The file is this module's whole device code, and this call renders only one
        # kernel plus the helpers it reaches. Keep what the file already holds as well, so
        # a module with two kernels keeps both.
        combined: str = merge_source(existing, source)
        if combined != existing:
            with open(path, "w", encoding="utf-8", newline="\n") as file:
                file.write(combined)
        return path

    if existing != source or existing is None:
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(source)
    return path
