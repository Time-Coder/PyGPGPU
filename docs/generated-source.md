# Generated source format

What a `@kernel` becomes on disk: where each file lands, what it declares, what it includes,
how the functions in it are named, and where its blank lines go. `pygpgpu/kernel/source_paths.py`
owns the paths and the assembly; `PyKernel._compile` in `decorator.py` decides the contents;
`translator.py` decides the shape of one function.

This is the reference for changing that pipeline. AGENTS.md points here rather than repeating
it, because a change to any one of these rules moves the generated output and the tests that
pin it (`tests/test_kernel_generated_layout.py`, `tests/test_kernel_generated_spacing.py`).

## One file per python module

A generated file belongs to a python *module*, not to a kernel:

    <cache>/pykernels/<module>.cu        a kernel's module       (opencl: .cl, hip: .hip)
    <cache>/pykernels/sub/<module>.cu    a module in a subpackage
    <cache>/pykernels/__/__/up.cu        a module above the kernel's own folder
    <cache>/pykernels/Lib/<module>.cu    a standard-library module
    <cache>/pykernels/Lib/site-packages/<module>.cu   an installed package's module

The cache folder is `__clcache__` / `__cucache__` / `__hipcache__`, beside the source. A kernel's
own module writes directly into `pykernels/` -- there is no per-kernel wrapper folder, because
the cache already sits beside the kernel's file.

`source_paths.source_path(kernel_file, source_file, backend)` derives all of that from two python
paths. A relative path that would climb above the kernel's folder is spelled with `__` for each
`..`, because a generated file never escapes the cache root.

A module's file holds its kernel *and* every device helper that kernel calls but that is defined
elsewhere in the same module. A second kernel from that module appends to the same file rather
than starting another one (`append_source`), and the definitions are de-duplicated by identity, so
calling one helper from two kernels writes it once.

## Cross-module references are one `#include`

A helper defined in *another* module is written to that module's own file, and the file that calls
it adds `#include "other_module.cu"` (path spelled relative to the referrer). `cpreprocessor.
macros_expand` expands the include **before** the backend compiler runs, so the final text the
compiler sees is flat: the layout is a property of the python source tree, not of one kernel. Each
file is still wrapped in its own guard, so a file reached from several others is read once.

## Declarations come first, then the includes

A generated file is laid out as:

1. the file guard (`#ifndef __<file>_<hash>__`);
2. the structure typedef block;
3. the forward-declaration block (see the next section);
4. the `#include` lines, in dependency order;
5. the device functions and the kernel, in dependency order;
6. `#endif`.

**The typedef block has to precede the includes**, because an included file's text may name those
types: `scene.cu`'s helper takes a `Ray`, so `Ray` must already be in scope where the `#include`
expands. `assemble_source` therefore keeps the block order the caller built instead of re-sorting
by "a line starting with `#` leads" -- that re-sort put the includes ahead of the declarations and
broke every cross-module helper.

**The forward-declaration block has to precede the includes too**, for the same order reason one
level down: an included file's functions call device functions whose *definitions* sit behind the
include -- the base-class helper a subclass override reaches is defined in the kernel's own file,
after the `#include` that pulls the override in. C refuses a call it has not seen declared
(`implicit declaration of function ... is invalid in OpenCL`, `identifier ... is undefined` on
CUDA/HIP), so every file a compilation writes carries one block declaring the signature of every
device function the compilation reached, spelled as the definition's own first line plus `;`.
Duplicate identical declarations are legal C, so the block repeating across a unit's files costs
nothing.

The includes follow the **same dependency order** as the definitions, because C needs a callee
before its caller: `scene.cu` calls `Sphere.hit`, so `#include "sphere.cu"` has to be read first.
`_called_modules` works out which other modules a file's own code calls into.

## The forward-declaration block is unioned, never replaced

A file is rewritten on every kernel call of its module, and each call renders only the callees
*that call* reached. Two kernels of one module can reach different helpers of another module, so
the block a later call writes holds fewer declarations than the file already carries -- replacing
the block would drop the earlier ones and break the first kernel's own calls. The block therefore
carries a marker comment as its first line, which gives the whole block one merge identity:
`_merge_definitions` unions the blocks by their declaration text and keeps the union where the
first block sat, and `_canonical_header` folds the blocks of several calls into one. A declaration
whose function has since left the file is dead text, which costs nothing.

## Structure typedefs belong to the whole compilation

The type block is collected from `translator.structure_dtypes()` plus every `device_cache` entry's
`structs`, and emitted **once** per file. Emitting it per function defines the same struct twice,
which C++ rejects; emitting it nowhere fails only when a helper is involved -- the ray-tracing
example compiled its own kernel and then failed on the helper, which is how that rule was found.

Each declaration is wrapped in its own guard (`#ifndef __<C name>__`), because two modules can name
one layout -- therefore one C type -- and whichever file is reached first declares it.

## A union is declared `union`, not `struct`

The dtype decides which keyword the block uses: `structs.is_union_dtype` reads the field **offsets**,
and a layout whose members all sit at offset zero is a union. Writing it as a `struct` would place
the second member after the first, so the device type would be as wide as the sum of the members
instead of the widest one -- and the host, which uploads the caller's own (narrower) value, would be
rejected with "Buffer size too small". Both ctypes dialects reach this: `ctypes.Union` and
`ctypes.LittleEndianUnion` (siblings of `ctypes.Structure`, not subclasses), and a hand-written
`np.dtype` with explicit zero offsets, which is the same layout described the same way.

    typedef union { int i; float f; } U_<hash>;      opencl
    union U_<hash> { int i; float f; };              cuda / hip
    typedef union U_<hash> U_<hash>;

The parsed type on the host side follows the same keyword, so the launch copies both directions
through a value of the right width.

## Names

| What | Shape | Where the class comes from |
|---|---|---|
| kernel, plain function | `name_<hash>` | -- |
| kernel that is a method | `Class_name_<hash>` | `__qualname__` |
| device function | `Class_method_<hash>` or `name_<hash>` | `__qualname__` |
| structure | `Class_<hash>` | the class that registered the layout first |
| file guard | `__<file>_<hash>__` | the file's own path |
| structure guard | `__<C name>__` | the C type name |

The hash covers the module plus the argument-type signature **and the rendered parameter list**,
which is what keeps two helpers of one name apart: a helper monomorphizes per call type (`blend`
over `float` and over `float3` are two device functions), and two modules can define the same
name. The rendered list matters because the generated signature carries more than the caller's
argument types -- the extents derived from the body, the outer-scope values, the shared buffers
-- so hashing the types alone left a body whose derived signature had changed under its old
name, and the file then held two declarations of it. **Nothing appends a number to settle a
collision** -- the hash is what does that, so a name never reads `blend_ab12_1`, and a kernel
name keeps its single `name_<hash>` shape.

A **method kernel's** hash also covers the class the launched *instance* belongs to. Its device
body is not the decorated body alone: a body that calls `self.<hook>(...)` compiles against the
hook the receiver's class supplies, and two subclasses of one `@kernel` supply different hooks
over the identical signature. Without the class in the hash -- and in the in-memory launch cache
key, which carries the same mark -- the second subclass to launch reused the first one's binary
and silently answered the first one's numbers.

## A generated file is invalidated by its source's digest, not only by its timestamp

`append_source` keeps a file whose contents describe the current python text and starts it over
otherwise. The first test was the mtime, which assumes a file's stamp moves forward when its text
changes; `git checkout` and `git stash pop` restore a file **and** set its mtime back, so a file
written from newer text kept looking fresh and its stale declaration survived beside the new
one. The digest of the python text a file came from is now recorded **beside** it, in
`<source>.source-digest`, and a mismatch invalidates it whichever way the timestamps point. A
sidecar rather than a comment inside the generated text: that text is a translation unit whose
blank lines `_split_source` splits on, so threading one more line through it would change how a
later write re-reads the file. An unchanged source keeps the file, so the append rule still
holds: a module's other kernels stay in place.

A method is named after the class it is *defined on*, never the class of the kernel that calls it:
`Ray.at` is `Ray_at`. A structure's C name is **pinned the first time it is asked for**
(`structs._SETTLED_NAMES`): a class can register after the dtype was already named, and a name that
changed afterwards would change the type signature of kernels already compiled against the old
spelling, which put two copies of one function into one file.

## The receiver is a method's first parameter

`Class_method(const Class* self, <the caller's arguments>, <self arrays>)`. Python spells the
receiver first and a device helper's signature already did, so all three places that decide the
order agree:

* `self.params` **is** the signature order, so the receiver is bound before the positional
  arguments;
* the call site (`_emit_device_call`) forwards the receiver first;
* the launch packs its values in that order -- receiver, then the caller's arguments, then the
  ndarray attributes.

The receiver's *type* keeps its slot after the positional argument types in `param_types`, which is
where the host already builds `self_types`; the two lists keep their own orders instead of one being
reordered to match the other.

## Blank lines

A generated file reads as source, so its spacing follows source conventions rather than the
emitter's convenience:

* **every closed block is followed by one blank line**, whatever nested level it closes at;
* **not** between `if` and `else` -- the two halves are one statement;
* **not** when the very next line closes the parent block (`for { if { ... } }`), where a blank line
  would separate two braces and nothing else;
* a run of `#include` lines is **one block** with one blank line after the last of them, not one
  blank line after each.

`translator._render_body` applies this to a body, and `_emit_element_loop` records its own
separator so a `for` loop's closing brace is followed by one too. A body whose last line closes a
block therefore still ends without a trailing blank, which is what lets the kernel's own closing
brace sit directly under the inner one.
