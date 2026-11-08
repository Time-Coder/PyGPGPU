"""The prelude tests, split out of `test_metal_codegen.py`.

The subject is prelude.
"""
from __future__ import annotations


from typing import Any
from pygpgpu.kernel.metal_codegen import (
    prelude,
)


def test_the_prelude_macro_body_is_parenthesised_as_a_whole() -> None:
    """The macro body is one parenthesised expression, so it composes anywhere.

    A conditional expression binds looser than `%`, and the folding the translator
    emits (`(gid % size) * stride + ...`) puts a macro expansion on the left of a
    `%`. Without the outer parentheses the conditional would swallow the rest of
    that expression and the work position would be wrong -- silently, on the
    device. The generated *body* was verified against a real C compiler evaluating
    the same fold (position 31 from a 8x4x1 grid at (7, 3, 0)); this test pins the
    parentheses that made it correct.
    """
    from pygpgpu.kernel.metal_codegen import prelude

    for line in prelude().splitlines():
        if not line.startswith("#define"):
            continue
        body: str = line.split(")", 1)[1].strip()
        # The whole body is wrapped: it opens with a parenthesised group and ends
        # by closing that same group, and the brackets balance.
        assert body.startswith("(") and body.endswith(")"), line
        assert body.count("(") == body.count(")"), line
        depth: int = 0
        for index, character in enumerate(body):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                # The outer group must not close before the very last character.
                assert depth != 0 or index == len(body) - 1, line


def test_the_prelude_is_one_blank_line_free_block() -> None:
    """The prelude must not contain a blank line, or the writer splits it apart.

    A generated file's header is the pieces between blank lines, and a piece whose
    first line is not `#`/`struct`/`typedef struct` is classified as a function
    definition (`source_paths._split_source`). A `using namespace metal;` that
    starts such a piece is dropped on the next write, and every macro then names an
    unqualified `uint3` -- an MSL compile error. Keeping the three parts on
    consecutive lines makes the whole prelude one piece that starts with `#`, so it
    dedupes by its exact text and survives every rewrite.
    """
    text = prelude()
    assert "" not in text.splitlines(), "a blank line would split the prelude"
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].startswith("#include <metal_stdlib>")
    assert lines[1] == "using namespace metal;"
    # Every remaining line is a macro.
    assert all(line.startswith("#define ") for line in lines[2:])


def test_writing_the_same_kernel_twice_is_byte_identical(tmp_path: Any) -> None:
    """Recompiling a kernel must not grow the file: one include, one using, one macro.

    `append_source` writes the header again on every compile of a module, so a
    header that does not dedupe accumulates a copy per call. The Metal prelude is
    the case that used to fail: its `using` line was split off and re-added.
    """
    from pygpgpu.kernel.source_paths import append_source

    path = str(tmp_path / "gen.metal")
    header = [prelude()]
    written: list = []
    for _ in range(3):
        append_source(path, header, ["kernel void k(device float* a) {}"],
                      __file__, source_digest="digest")
        written.append(open(path, encoding="utf-8").read())

    for source in written[1:]:
        assert source.count("#include <metal_stdlib>") == 1
        assert source.count("using namespace metal;") == 1
        assert source.count("#define get_global_id(dim)") == 1
    # The same definition is written three times; the file must not grow after the first.
    assert written[1] == written[2]
    assert written[0] == written[1]


def test_a_module_with_two_kernels_keeps_one_prelude(tmp_path: Any) -> None:
    """Writing a second kernel must not duplicate the prelude in the shared file.

    One python module owns one generated file and each kernel is compiled on its own
    call, so the header is re-written every time. Two kernels therefore means two
    header writes, and the prelude has to dedupe by its exact text -- which is why
    it is one blank-line-free block (see the test above).
    """
    from pygpgpu.kernel.source_paths import append_source

    source_file = str(__file__)
    path = str(tmp_path / "multi.metal")
    # Two different kernels, each written twice, in one file.
    for _ in range(2):
        for body in ("kernel void one(device float* a) {}", "kernel void two(device float* a) {}"):
            append_source(path, [prelude()], [body], source_file, source_digest="d")
    written = open(path, encoding="utf-8").read()

    assert written.count("#include <metal_stdlib>") == 1
    assert written.count("using namespace metal;") == 1
    assert written.count("#define get_global_id(dim)") == 1
    assert written.count("kernel void one") == 1
    assert written.count("kernel void two") == 1
