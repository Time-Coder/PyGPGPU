"""The vocabulary tests, split out of `test_metal_launch_path.py`.

The subject is vocabulary.
"""
from __future__ import annotations



def test_every_science_vocabulary_entry_has_a_metal_spelling() -> None:
    """Metal must carry a spelling for every entry, or refuse by name -- never KeyError.

    The tables are keyed by backend, so a missing entry is a `KeyError` raised from
    inside the translation. Either a backend carries a spelling or the translator
    refuses the call with a message naming the backends that do: those are the only
    two acceptable outcomes.
    """
    from pygpgpu.kernel.device_builtins import SHARED_MATH_BUILTINS, SHARED_MATH_TEMPLATES

    for name, table in list(SHARED_MATH_BUILTINS.items()) + list(SHARED_MATH_TEMPLATES.items()):
        assert "metal" in table, f"{name} has no metal spelling and would KeyError"
        assert table["metal"], f"{name}'s metal spelling is empty"
    # The two tables are disjoint: a name lives in exactly one, so the lookup order
    # cannot change what is emitted.
    assert not (set(SHARED_MATH_BUILTINS) & set(SHARED_MATH_TEMPLATES))


def test_the_fast_and_exact_families_are_refused_on_metal() -> None:
    """The CUDA-only vocabulary has no Metal counterpart, and the refusal says so."""
    from pygpgpu.kernel.device_builtins import CUDA_ONLY_BUILTINS, CUDA_ONLY_CONVERSIONS

    # Every CUDA-only entry is spelled for cuda and hip, and for nothing else.
    for name, table in list(CUDA_ONLY_BUILTINS.items()) + list(CUDA_ONLY_CONVERSIONS.items()):
        assert "metal" not in table, (
            f"{name} claims a Metal spelling; if Metal really has this intrinsic, "
            f"move it into the shared tables instead")
        assert {"cuda", "hip"} <= set(table), f"{name} is not spelled for cuda and hip"


def test_a_shared_math_name_emits_its_metal_spelling() -> None:
    """What `_shared_math_call` emits is the table's `metal` entry, verbatim.

    The two shapes are pinned separately because they are genuinely different: a
    callee entry becomes `name(args)`, and a template entry is formatted with the
    single argument. Metal's `native_recip` is the second kind; everything else is
    the first.
    """
    from pygpgpu.kernel.device_builtins import SHARED_MATH_BUILTINS, SHARED_MATH_TEMPLATES

    for name, table in SHARED_MATH_BUILTINS.items():
        spelling = table["metal"]
        # A callee spelling is a bare identifier path, never an expression.
        assert "(" not in spelling and " " not in spelling, (
            f"{name}'s metal spelling {spelling!r} is an expression, so it belongs in "
            f"SHARED_MATH_TEMPLATES")
    for name, table in SHARED_MATH_TEMPLATES.items():
        # A template must consume the argument placeholder exactly once per use.
        assert "{code}" in table["metal"], (
            f"{name}'s metal template never consumes its argument")
        rendered = table["metal"].format(code="x")
        assert "x" in rendered
