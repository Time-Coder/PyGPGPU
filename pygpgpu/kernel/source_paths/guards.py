# The include guard, the include line and how a piece of source is classified.
from __future__ import annotations
from .layout import _sanitize
import hashlib
import os
from typing import List



QUAT_DEFINE_PREFIX: str = "#define QUAT_"


# The first line of the block that forward-declares the device functions one generated file
# calls but does not define. It is a comment, so every compiler ignores it, and it is what
# gives the block an identity of its own: `merge._definition_identity` answers it, so the
# block is recognised, kept first and *unioned* rather than appended beside an older copy.
PROTOTYPE_MARKER: str = "/* pygpgpu: device functions called here but defined elsewhere */"



# The folder a cache keeps its device headers in; the same spelling `complex.py` and
# `quaternion.py` use.
INCLUDE_FOLDER: str = "include"



# The shared header the *synthesized* device helpers live in (see `writing.append_helpers`).
# Unlike `complex.h` / `quat.h` it is not a file the package ships: its body is generated,
# one definition per helper shape, merged the way a module's own file merges its kernels.
HELPERS_HEADER_NAME: str = "device_helpers.h"



def guard_macro(path: str) -> str:
    """The include guard a generated file's own path gives it.

    Two generated files can declare the same structure (a `Ray` in two modules has one
    layout, therefore one C name), so a source that reaches both would define the type
    twice -- which C++ rejects. A guard per file fixes exactly that: the second inclusion
    is skipped whole, and its declarations come from the first.

    The hash covers the file's path **inside the cache tree**, not its absolute path. The
    guard has to be the same for a cache that was moved: a tree copied into another
    directory (or shipped inside a packaged application) rewrites every absolute path, and a
    guard that changed with it would make the new text differ from the old -- so the
    generated source would no longer match its own digest, and the parse and binary caches
    would both miss on every start. What the guard has to separate is two *modules*, and
    their paths inside the tree already do that.
    """
    relative: str = _tree_relative_path(path)
    digest: str = hashlib.md5(
        os.path.normcase(os.path.abspath(path)).encode("utf-8")).hexdigest()[:12]
    if relative:
        digest = hashlib.md5(relative.encode("utf-8")).hexdigest()[:12]
    # `__<file>_<hash>__`: the file's own name is the readable half, the hash keeps two
    # same-named modules (both `scene.cu`, say) apart, and the trailing underscores mark
    # the end the way a leading pair marks the start.
    stem: str = os.path.splitext(os.path.basename(path))[0] or "source"
    return f"__{_sanitize(stem)}_{digest}__"



#: The cache folder a generated file sits in, as a marker. A generated source always lives
#: below one of these, so everything after it is the file's path inside the cache tree.
_CACHE_MARKERS: tuple = ("__cucache__/", "__clcache__/", "__hipcache__/", "__metalcache__/")


def _tree_relative_path(path: str) -> str:
    """The part of a generated file's path below its cache folder, with `/` separators.

    Empty when the path does not name a cache folder, which keeps `guard_macro` usable for a
    caller that hands it an arbitrary file (a test's temporary source).
    """
    normalized: str = os.path.abspath(path).replace("\\", "/")
    for marker in _CACHE_MARKERS:
        position: int = normalized.find("/" + marker)
        if position != -1:
            return normalized[position + 1:]
    return ""



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



def _is_fixed_header_include(line: str) -> bool:
    """True for the `#include` of a header this package installs into the cache.

    Those live in the cache's `include/` folder (see `complex.include_line` /
    `quaternion.include_line`), which is what the quoted path names.
    """
    return line.lstrip().startswith(f'#include "{INCLUDE_FOLDER}/')
