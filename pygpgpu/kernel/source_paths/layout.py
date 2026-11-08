# Where a generated file lives: the cache folder and the module path under it.
from __future__ import annotations
import os
import site
from typing import Dict, List, Optional, Tuple



# The extension a generated source needs, per backend.
EXTENSIONS: Dict[str, str] = {"opencl": "cl", "cuda": "cu", "hip": "hip", "metal": "metal"}


EXTENSIONS: Dict[str, str] = {"opencl": "cl", "cuda": "cu", "hip": "hip", "metal": "metal"}
# The cache folder a backend writes into.
CACHE_FOLDERS: Dict[str, str] = {"opencl": "__clcache__", "cuda": "__cucache__",
                                 "hip": "__hipcache__", "metal": "__metalcache__"}



# The directory name for a python file from the standard library, and for one from an
# installed package: both mirror their own path below these roots.
LIB_FOLDER: str = "Lib"


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
    # Through the facade, not through this module: `cache_root` is a module attribute a
    # caller may redirect, and a local name would keep pointing at the original definition.
    from . import cache_root
    return os.path.join(cache_root(kernel_file, backend), directory,
                        f"{_sanitize(stem)}.{extension}")



def cache_root(kernel_file: str, backend: str) -> str:
    """The `pykernels` folder a kernel's generated sources live in."""
    return os.path.join(source_directory_of(kernel_file), CACHE_FOLDERS[backend],
                        "pykernels")
