"""Metal Shading Language code generation.

Metal's device source is close enough to OpenCL C that the translator's existing
OpenCL emission is the right starting point -- kernels, pointers into buffers,
atomics, the same vector types -- and far enough that a blind copy would not
compile. The differences are few and each is one fact, so they live here as one
module rather than as a `backend == "metal"` branch at every emission site:

* an entry point is `kernel void f(...)`, not `__kernel void f(...)`;
* an address space is `device`/`threadgroup`, not `__global`/`__local`;
* the work-item position arrives as a **kernel parameter** carrying a
  `[[thread_position_in_grid]]` attribute, not as a `get_global_id()` call;
* a work-group barrier is `threadgroup_barrier(mem_flags::mem_threadgroup)`.

The last difference is the interesting one. The shared body text calls
`get_global_id(0)` / `get_local_id(1)` / `get_num_groups(2)`, and MSL has no such
global functions -- those numbers only exist as attribute-decorated kernel
parameters. A **function-like macro** solves it with no change to the body at
all: every generated kernel declares parameters of fixed names, so
`#define get_global_id(dim)` can expand to that parameter's component. The body
then compiles as MSL exactly as it was written for OpenCL, and the only
Metal-specific text is the prologue and the signature.

Every attribute name and spelling here follows Metal Shading Language 3.x; the
macros are exercised by the `metal` cases in `tests/`, which need a Mac (see
AGENTS.md).
"""
from __future__ import annotations

from typing import List, Tuple

# The parameters every generated Metal kernel declares, with the attribute that
# supplies each. The names are fixed because the macros below expand onto them --
# a kernel whose signature spelled these differently would compile to a body
# referring to undeclared identifiers. The declaration order is (type, name,
# attribute) as MSL spells it: the type first, the attribute after the name.
PROLOGUE_PARAMETERS:Tuple[Tuple[str, str, str], ...] = (
    ("uint3", "_mtl_thread_position", "thread_position_in_grid"),
    ("uint3", "_mtl_thread_index", "thread_index_in_threadgroup"),
    ("uint3", "_mtl_threadgroup_position", "threadgroup_position_in_grid"),
    ("uint3", "_mtl_threads_per_threadgroup", "threads_per_threadgroup"),
    ("uint3", "_mtl_threads_per_grid", "threads_per_grid"),
    ("uint3", "_mtl_threadgroups_per_grid", "threadgroups_per_grid"),
)

# The OpenCL spelling a shared body calls -> the prologue parameter it reads.
_INDEX_SOURCES:Tuple[Tuple[str, str], ...] = (
    ("get_global_id", "_mtl_thread_position"),
    ("get_local_id", "_mtl_thread_index"),
    ("get_group_id", "_mtl_threadgroup_position"),
    ("get_global_size", "_mtl_threads_per_grid"),
    ("get_local_size", "_mtl_threads_per_threadgroup"),
    ("get_num_groups", "_mtl_threadgroups_per_grid"),
)


def prelude()->str:
    """The include, the `using` and the macros, as ONE blank-line-free block.

    Three constraints decide the shape:
    (1) MSL needs `#include <metal_stdlib>` and `using namespace metal;` before the kernel
        signature, because that signature names `uint3` unqualified;
    (2) a generated file's header is split on BLANK LINES and every piece whose first
        line is not `#`/`struct`/`typedef struct` is classified as a function definition
        (`source_paths._split_source`), so a `using` line must never start a piece of its
        own;
    (3) `append_source` writes the header again on every compile of the same module, so
        the block has to dedupe by its exact text (`_canonical_header` keeps a mixed
        block whole and skips an identical one).
    Emitting the three parts with no blank line between them satisfies all three: it is
    one piece, it starts with `#`, and re-adding it is a no-op.
    """
    lines:List[str] = [
        "#include <metal_stdlib>",
        "using namespace metal;",
    ]
    for opencl_name, parameter in _INDEX_SOURCES:
        components:str = ("((int)(dim) == 0 ? (int)%s.x : ((int)(dim) == 1 ? "
                          "(int)%s.y : (int)%s.z))" % (parameter, parameter, parameter))
        lines.append(f"#define {opencl_name}(dim) ({components})")
    return "\n".join(lines) + "\n"


def kernel_prefix()->str:
    """The qualifier *and return type* an entry point carries.

    MSL writes the return type in the declaration (`kernel void f(...)`), so the
    caller must not append another `void` after this.
    """
    return "kernel void"


def prologue_parameters()->List[str]:
    """The attribute parameters an MSL kernel signature carries, as C text.

    MSL spells an attributed parameter ``type name [[attribute]]`` -- the attribute
    follows the name, not the type.
    """
    return [f"{type_text} {name} [[{attribute}]]"
            for type_text, name, attribute in PROLOGUE_PARAMETERS]


def address_space_qualifier(opencl_word:str)->str:
    """OpenCL's address-space word as Metal spells it."""
    return {
        "__global": "device",
        "__local": "threadgroup",
        "__constant": "constant",
        "__private": "thread",
    }.get(opencl_word, opencl_word)


def barrier()->str:
    """A work-group barrier.

    The threadgroup space is what a `zeros_per_group` buffer lives in, so its
    writes have to become visible across the group -- the same reason the OpenCL
    spelling names the local fence. The device fence is added because the array
    arguments are written through the same barrier.
    """
    return "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)"
