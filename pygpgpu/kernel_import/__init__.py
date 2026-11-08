"""kernel_import -- import one module name from same-named `.cl`/`.cu`/`.hip` sources.

A hand-written kernel is written once per backend, so three backends of one kernel would
otherwise need three file names and three imports. This hook instead looks for ALL of
`<stem>.cl`, `<stem>.cu` and `<stem>.hip` in the same place and publishes ONE module named
`<stem>`:

    vecops.cl    -> opencl
    vecops.cu    -> cuda
    vecops.hip   -> hip
    from kernel_import import vecops

Every kernel in the sources becomes a module attribute used exactly like a `@kernel`
function: no bracket runs on the default device, `["cuda"]` / `["opencl:1"]` names a
backend or device, `[(blocks, threads)]` states the work size -- in either order. The same
kernel name declared with the same parameter list in several sources is ONE function whose
implementations differ per backend; declared with different parameter lists it is a set of
overloads. A bracket naming a backend that has no implementation for the name raises
`KeyError` at the bracket, before any argument is looked at.

Nothing here goes through a backend's import hook or its bracket-style wrapper: each
backend is reached through its own `ProgramParser` (metadata) and `Program`/`Kernel`
objects, which is the layer a launch is built on anyway.

A machine that lacks one of the runtimes is not a problem for the import: the driver layer
records a failed runtime load instead of failing its own import, so every source is parsed
and the failure surfaces -- as the driver's own error -- only when a kernel on that backend
is actually selected."""
# This module is the facade of a split: the hook, the wrapper it builds, the source
# loading and the scalar tables each moved to the module named after their subject,
# and every name it used to publish stays importable from here.
from __future__ import annotations


from .tables import (
    _BACKEND_EXTENSIONS,
    _METAL_SCALAR_NAMES,
    _METAL_TABLES,
    _METAL_VECTOR_WIDTHS,
    _PARSER_MODULES,
    _QUALIFIER_WORDS,
    _metal_scalar_tables,
    _scalar_tables,
    _signature_text,
    normalized_signature,
)
from .source import (
    _BackendSource,
    _source_paths_for,
)
from .tables import _work_size_item
from .wrapper import (
    KernelImportWrapper,
    _MergedModuleLoader,
)
from .hook import (
    _KernelImportFinder,
    kernel_import,
)
