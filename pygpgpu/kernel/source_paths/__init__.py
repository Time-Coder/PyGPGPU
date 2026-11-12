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
be noise."""
# This module is the facade of a split: every name it used to define is re-exported
# from the part module that now owns it, so `source_paths.<name>` and
# `from pygpgpu.kernel.source_paths import <name>` keep working unchanged.
from __future__ import annotations


from .layout import (
    CACHE_FOLDERS,
    EXTENSIONS,
    LIB_FOLDER,
    SITE_PACKAGES_FOLDER,
    _is_within,
    _library_relative,
    _relative_segments,
    _sanitize,
    _site_packages_directories,
    _standard_library_directory,
    cache_root,
    module_directory,
    package_source_files,
    source_directory_of,
    source_path,
)
from .guards import (
    HELPERS_HEADER_NAME,
    INCLUDE_FOLDER,
    PROTOTYPE_MARKER,
    QUAT_DEFINE_PREFIX,
    _is_file_guard_line,
    _is_fixed_header_include,
    _is_include_only,
    _strip_file_guard,
    guard_macro,
    guard_source,
    include_line,
)
from .prototypes import (
    compilation_prototypes,
    merge_prototype_blocks,
    prototype_block,
    prototype_prototypes,
)
from .merge import (
    _CONTROL_KEYWORDS,
    _canonical_header,
    _definition_identity,
    _merge_definitions,
    _split_source,
    assemble_source,
    merge_source,
)
from .locks import (
    _FILE_LOCK_POLL,
    _FILE_LOCK_TIMEOUT,
    _MKDIR_ATTEMPTS,
    _SOURCE_WRITE_POLL,
    _SOURCE_WRITE_TIMEOUT,
    _WRITE_LOCKS,
    _WRITE_LOCKS_GUARD,
    _ensure_directory,
    _file_lock,
    _file_lock_path,
    _file_lock_state,
    _path_lock,
    write_lock,
)
from .writing import (
    DIGEST_SUFFIX,
    _append_source_locked,
    _stored_digest,
    _write_digest,
    _write_source_file,
    _write_source_locked,
    append_helpers,
    append_source,
    digest_sidecar,
    python_source_digest,
    rewrite_source,
    write_source,
)
