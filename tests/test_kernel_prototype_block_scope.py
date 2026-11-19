"""The forward-declaration block names only what each file's own code calls.

One compilation used to write the same block into every file it touched, declaring
*every* callee the whole unit reached: the ray-tracing example's `utils.cu` carried
sixteen declarations while its own code called three functions, and the block named
the very kernel whose file includes this one. The block is per file now -- a file
declares the callees its own definitions call and do not define themselves -- and the
merge narrows the union it holds, so a declaration a package-side regeneration stops
spelling is dropped from the file instead of surviving forever.
"""
from __future__ import annotations

from pygpgpu.kernel.source_paths.merge import merge_source
from pygpgpu.kernel.source_paths.prototypes import (
    compilation_prototypes,
    prototype_block,
)


# A two-file unit shaped like the ray-tracing example: the kernel's file holds the
# kernel and one helper; the foreign file holds the rand chain that calls the
# synthesized helpers (which live in the shared header, not in either file).
KERNEL_CODE = "__global__ void k(float* out)\n{\n    out[0] = rand_vector3f_hemisphere_5e1f46e4(a, 1, 2);\n}"
HEMISPHERE_CODE = (
    "__device__ float3 rand_vector3f_hemisphere_5e1f46e4(float3 normal, int tid, int seed)\n"
    "{\n"
    "    float3 unit_vec = rand_unit_vector3f_51b886ab(tid, seed);\n"
    "    return vec_neg_float3(unit_vec);\n"
    "}"
)
UNIT_CODE = (
    "__device__ float3 rand_unit_vector3f_51b886ab(int tid, int seed)\n"
    "{\n"
    "    len_result = vec_length_float3(result);\n"
    "    return vec_div_float3_float(result, len_result);\n"
    "}"
)
RAND_CODE = "__device__ float rand_51b886ab(int tid, int seed)\n{\n    return 0;\n}"
HELPERS = [
    "__device__ float3 vec_dot_float3_float3(float3 a, float3 b)\n{\n    return a.x;\n}",
    "__device__ float3 vec_neg_float3(float3 a)\n{\n    return a;\n}",
    "__device__ float vec_length_float3(float3 a)\n{\n    return 0;\n}",
    "__device__ float3 vec_div_float3_float(float3 a, float b)\n{\n    return a;\n}",
]
ENTRIES = [{"code": code} for code in (HEMISPHERE_CODE, UNIT_CODE, RAND_CODE)]


def test_the_kernel_file_declares_what_it_calls_across_the_boundary():
    """The kernel's file declares its own chain of cross-boundary calls.

    The kernel calls the hemisphere helper (its own line first), whose definition here
    calls the foreign module's `rand_unit_vector3f` and the shared `vec_neg` helper --
    each is a call this file's own definitions spell, so all three are declared. The
    helpers the *foreign* file's definitions reach (`vec_length`, `vec_div`) are not
    this file's calls and stay out.
    """
    lines = compilation_prototypes(ENTRIES, HELPERS,
                                   own_codes=[KERNEL_CODE, HEMISPHERE_CODE])
    assert lines == [
        "__device__ float3 rand_vector3f_hemisphere_5e1f46e4(float3 normal, int tid, int seed);",
        "__device__ float3 rand_unit_vector3f_51b886ab(int tid, int seed);",
        "__device__ float3 vec_neg_float3(float3 a);",
    ]


def test_a_foreign_file_declares_its_own_calls():
    """The rand chain calls two synthesized helpers; the sibling's own head line is a
    definition and not a call, so the sibling is not declared."""
    lines = compilation_prototypes(ENTRIES, HELPERS, own_codes=[UNIT_CODE, RAND_CODE])
    assert lines == ["__device__ float vec_length_float3(float3 a);",
                     "__device__ float3 vec_div_float3_float(float3 a, float b);"]


def test_a_call_cycle_stays_declared():
    """Two mutually recursive helpers need the forward declarations to compile.

    The emitter orders callees before callers, but a call cycle has no leaf: `odd`
    reads `even`, whose definition follows, and C refuses the call without the line.
    This is why a name the file defines itself is still declared when a call spells
    it (see compilation_prototypes).
    """
    odd_code = "__device__ int odd_200096e5(int n)\n{\n    return even_200096e5((n - 1));\n}"
    even_code = "__device__ int even_200096e5(int n)\n{\n    return odd_200096e5((n - 1));\n}"
    kernel_code = "__global__ void k(int* out, int n)\n{\n    out[0] = even_200096e5(n);\n}"
    entries = [{"code": code} for code in (odd_code, even_code)]
    lines = compilation_prototypes(entries, [], own_codes=[odd_code, even_code, kernel_code])
    assert "__device__ int odd_200096e5(int n);" in lines
    assert "__device__ int even_200096e5(int n);" in lines


def test_a_file_whose_code_calls_nothing_known_carries_no_block():
    """`rand` calls nothing the compilation reached: an empty list, so the file gets
    no block at all. (Its own head line is a definition, not a call of itself.)"""
    assert compilation_prototypes(ENTRIES, HELPERS, own_codes=[RAND_CODE]) == []


def test_the_merge_drops_a_declaration_the_file_stopped_calling():
    """The union narrows: an old block's dead line does not survive the next write.

    The file's older block also declared a third-module function its remaining code
    never calls -- a declaration the whole-unit render once stacked into every file.
    The merge removes that one and keeps the two its definitions still spell.
    """
    existing = "\n\n".join([
        "#ifndef __x__\n#define __x__",
        prototype_block([
            "__device__ float vec_length_float3(float3 a);",
            "__device__ float3 vec_div_float3_float(float3 a, float b);",
            "__device__ float third_module_helper(int x);",
        ]),
        UNIT_CODE,
        RAND_CODE,
        "#endif",
    ])
    addition = "\n\n".join([UNIT_CODE, RAND_CODE])  # no block: nothing cross-file now
    merged = merge_source(existing, addition)
    assert "__device__ float third_module_helper(int x);" not in merged
    assert "__device__ float vec_length_float3(float3 a);" in merged
    assert "__device__ float3 vec_div_float3_float(float3 a, float b);" in merged


def test_the_merge_drops_every_declaration_when_no_call_stays():
    """A regeneration that removes the only cross-file call empties the block."""
    existing = "\n\n".join([
        "#ifndef __x__\n#define __x__",
        prototype_block(["__device__ float third_module_helper(int x);"]),
        UNIT_CODE,
        "#endif",
    ])
    addition = UNIT_CODE
    merged = merge_source(existing, addition)
    assert "third_module_helper" not in merged


def test_the_merge_keeps_a_declaration_a_call_still_needs():
    """A file whose older block declared a helper its code still calls keeps it."""
    existing = "\n\n".join([
        "#ifndef __x__\n#define __x__",
        prototype_block(["__device__ float vec_length_float3(float3 a);",
                         "__device__ float3 vec_div_float3_float(float3 a, float b);"]),
        UNIT_CODE,
        RAND_CODE,
        "#endif",
    ])
    addition = "\n\n".join([UNIT_CODE, RAND_CODE])
    merged = merge_source(existing, addition)
    assert "__device__ float vec_length_float3(float3 a);" in merged
    assert "__device__ float3 vec_div_float3_float(float3 a, float b);" in merged
