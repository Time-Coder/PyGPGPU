"""Known-failure regressions about the object side: overrides, spells, and the vectors.

Every test here fails today, on purpose: they are the specification for the fixes, written
as failing tests first (TDD). Each one names the repro that found it and what the right
answer is, and none of the answers invents a new rule -- each is what python (or numpy,
where the interpreter runs numpy) already computes for the same source. The kernels live
one module per bug in `known_failures/`, because the generated source of one python module
is one translation unit: a kernel whose bug is a compile error would otherwise take every
other kernel in the module down with it. The arithmetic groups live in
`test_kernel_known_failures.py`; this file holds the three object-shaped groups:

* **a subclass override and its helpers**: a method a subclass overrides may not call a
  helper of the base class, nor a module-level function -- the generated source calls it
  without defining it -- and two subclasses of one `@kernel` share the first one's
  compiled binary, so the second silently answers the first one's colours.
* **python spellings the translator refused outright**: `super().method(...)` and
  `Base.method(self, ...)` from an override were refused by name (`unknown function 'super'`,
  `unknown name 'self'`), although both compile to the base helper call the base's own body
  already produces; and `raise` in a device method was a blanket "unsupported statement"
  instead of the abort it means on a device.
* **the pygf vectors on the store side**: a shader could not write one component of a local
  `color3f`/`float2` (`c.x = ...`), nor through a swizzle -- "member stores need a structure
  value, and float3 is not one" -- although reads, swizzle reads, whole-vector stores and a
  vector self member all work, and the host mutates the same objects freely.
* **a damaged generated file**: a file that lost text any digest cannot see (a previous
  package version's shape, a crashed write) stayed broken for the life of the cache.
"""
from __future__ import annotations

import numpy as np

from known_failures.damaged_generated_file import Base as DamagedFileBase
from known_failures.damaged_generated_file_driver import generated_files, marker_present
from known_failures.qualified_call_with_receiver import CallsQualified
from known_failures.raise_statement import RaisingHelper
from known_failures.subclass_calls_base_helper import Base as SubclassBaseHelperBase
from known_failures.subclass_calls_module_helper import Base as SubclassModuleHelperBase
from known_failures.subclass_calls_module_helper import scaled
from known_failures.subclass_kernel_identity import Base as SubclassIdentityBase
from known_failures.super_call import CallsSuper
from known_failures.vector_component_store import ComponentStore as _ComponentStore

# ---------------------------------------------------------------------------
# Group 5 -- a subclass override and the helpers it calls
#
# Found by writing `examples/fractal/fractal_viewer.py`: a base class owns the @kernel and
# its body calls an overridable hook, and three viewers in other modules write that hook.
# Two of the three compile; the one whose override reaches a helper of the base does not.
# ---------------------------------------------------------------------------


def _sample_grid(factor: float) -> np.ndarray:
    """What `sample(i, j)` answers when it is `factor * (i + j + 1)`."""
    rows = np.arange(2, dtype=np.float32)[:, None]
    columns = np.arange(3, dtype=np.float32)[None, :]
    return factor * (rows + columns + 1.0)


class _UsesInheritedHelper(SubclassBaseHelperBase):
    """The override colours through a helper it inherits, the way the examples do."""

    def sample(self, i: int, j: int) -> np.float32:
        return self.scale(np.float32(i + j + 1))


def test_a_subclass_override_can_call_an_inherited_helper(gpu_backend):
    """`self.scale(...)` from an override must compile, exactly as from the kernel body.

    The generated source calls the base's device function and never defines it, so the module
    is refused with `identifier "Base_scale_<hash>" is undefined`. Nothing about the call is
    ambiguous -- the receiver is the instance, the class is known at translation time -- and
    the same helper is emitted correctly when the *kernel body* calls it directly, so the walk
    that collects device functions simply has to reach through an override too. The expected
    numbers are python's: `sample(i, j)` is `2 * (i + j + 1)`.
    """
    viewer = _UsesInheritedHelper()
    viewer.draw[gpu_backend]()
    np.testing.assert_allclose(viewer.image, _sample_grid(2.0))


class _UsesModuleHelper(SubclassModuleHelperBase):
    """The override colours through a module-level function."""

    def sample(self, i: int, j: int) -> np.float32:
        return scaled(np.float32(i + j + 1))


def test_a_subclass_override_can_call_a_module_level_function(gpu_backend):
    """A module-level device helper is reachable from a kernel body and from an override alike.

    Same failure as the inherited helper above (`identifier "scaled_<hash>" is undefined`), and
    the same reason: the helpers an override reaches are not walked for emission. The expected
    numbers are python's: `sample(i, j)` is `3 * (i + j + 1)`.
    """
    viewer = _UsesModuleHelper()
    viewer.draw[gpu_backend]()
    np.testing.assert_allclose(viewer.image, _sample_grid(3.0))


class _AnswersOne(SubclassIdentityBase):
    """The first subclass to launch."""

    def sample(self, i: int, j: int) -> np.float32:
        return np.float32(1)


class _AnswersTwo(SubclassIdentityBase):
    """A second subclass, whose hook answers differently."""

    def sample(self, i: int, j: int) -> np.float32:
        return np.float32(2)


def test_two_subclasses_of_one_kernel_keep_their_own_hook(gpu_backend):
    """The second subclass must not be served the first one's device code.

    The compiled-launch cache is keyed by backend, device, the *argument* signature and the
    type map; the class whose hook the kernel inlined is in none of them, so the second launch
    reuses the first binary and every pixel comes back 1 instead of 2. That is the worst kind
    of failure -- a wrong answer with no error -- and it is what a script showing two viewers
    of one base class hits. Whatever supplies the identity, two different hooks must not share
    one cached launch.
    """
    first = _AnswersOne()
    second = _AnswersTwo()
    first.draw[gpu_backend]()
    second.draw[gpu_backend]()
    np.testing.assert_allclose(first.image, 1.0)
    np.testing.assert_allclose(second.image, 2.0)


# ---------------------------------------------------------------------------
# Group 6 -- python spellings the translator refuses outright
#
# Found while writing the same fractal viewer: an overridable hook wants to reach its base
# implementation (the two spellings python gives for that) and to guard itself with `raise`.
# All three are refused at translation time with a blanket reason, although each names an
# emission the translator already knows how to produce for a plain `self.` call.
# ---------------------------------------------------------------------------


def test_super_reaches_the_base_implementation(gpu_backend):
    """`super().sample(i, j)` from an override is the base helper called on the receiver.

    Today: `SyntaxError: unknown function 'super'`. The base implementation is an ordinary
    device helper like any other -- the base's own body compiles `self.sample(i, j)` to a call
    to it -- so the super call has to compile to the same emission. The expected numbers are
    python's: the base answers `2 * (i + j + 1)` and the override adds one.
    """
    viewer = CallsSuper()
    viewer.draw[gpu_backend]()
    np.testing.assert_allclose(viewer.image, _sample_grid(2.0) + 1.0)


def test_a_class_qualified_call_may_pass_the_receiver(gpu_backend):
    """`Base.sample(self, i, j)` is the same base helper called on the receiver.

    Today: `SyntaxError: unknown name 'self' ...`. This is the spelling python gives when the
    override is one expression away from the base answer and `super()` does not fit. The call
    site knows the class and the instance, so the emission is the one the base's own body
    already produces. The expected numbers are python's: the base answers
    `2 * (i + j + 1)` and the override adds one.
    """
    viewer = CallsQualified()
    viewer.draw[gpu_backend]()
    np.testing.assert_allclose(viewer.image, _sample_grid(2.0) + 1.0)


def test_a_device_helper_may_raise(gpu_backend):
    """`raise` in a device method guards the input, exactly as in any python helper.

    The refusal ("unsupported statement: Raise") is gone: a device work item cannot unwind
    into python, so a raise compiles to what it means on a device -- an abort of the whole
    launch. CUDA's NVRTC takes the PTX trap (`asm("trap;")`), HIP's clang takes
    `__builtin_trap()` (ROCm's HIP headers declare no trap function, and inline PTX would
    be rejected on AMD), and the interpreter backend raises the real exception, running
    python as it does. OpenCL C and MSL have no portable abort, so there the statement is
    emitted as a comment naming the limitation -- the one spelling that compiles everywhere.
    This launch never reaches the guard (`count=2 >= 0`), so the arithmetic answers python's
    numbers either way; the guard's firing behavior is what each backend's abort decides.
    """
    viewer = RaisingHelper(2)
    viewer.draw[gpu_backend]()
    # count >= 0, so the guard never fires and the arithmetic answers python's numbers.
    np.testing.assert_allclose(viewer.image, [[0.0, 2.0]])


# ---------------------------------------------------------------------------
# Group 7 -- the pygf vectors, found by writing `examples/shadertoy/`
#
# A shader assembles a colour one component at a time and reads its uniforms as vectors;
# both spellings are ordinary GLSL. Reading works -- swizzles, component reads, whole-vector
# stores, and a vector as a self member (pinned passing in
# tests/test_kernel_subclass_override.py) -- but a *component store* is refused, and the
# refusal names the gap: member stores accept "a structure value", and the pygf vectors are
# not recognised as one.
# ---------------------------------------------------------------------------


def test_a_vector_component_may_be_stored(gpu_backend):
    """`c.x = uv.x` on a local `color3f` is the component write the record layout already has.

    Today: `SyntaxError: cannot assign to 'c.x': member stores need a structure value, and
    float3 is not one`. The same refusal answers a swizzle store (`c.xy = uv`). The host
    mutates the very same object freely (`v.x = 9.0` on a `float2`), and the object-argument
    path already stores through plain-object members, so the store machinery exists; what is
    missing is the vector case in whatever decides "is this a structure value". The expected
    numbers are python's: the colour is `(uv.x, uv.y, 0)`.
    """
    viewer = _ComponentStore()
    viewer.draw[gpu_backend]()
    expected = np.zeros((2, 3, 3), dtype=np.uint8)
    for i in range(2):
        for j in range(3):
            expected[i, j] = (255 * ((j + 0.5) / 3.0), 255 * ((i + 0.5) / 2.0), 0)
    np.testing.assert_allclose(viewer.image, expected)


# ---------------------------------------------------------------------------
# Group 8 -- a damaged generated file is never repaired
#
# Found live, twice in one day: a launch failed with `Unresolved extern function` /
# `identifier "..." is undefined`, the run passed after hand-deleting `__*cache__`, and no
# code had changed in between. The freshness sidecar decides "does this file describe
# today's python source?" -- and nothing else. A file that lost text any other way (written
# by a previous package version whose generated shape changed, a crashed write, a truncated
# copy) keeps a digest that still matches, so the damage is permanent for the life of the
# cache, and the user the contract addresses never deletes the cache by hand.
# ---------------------------------------------------------------------------


class _AnswersOneDamaged(DamagedFileBase):
    """The first subclass to launch."""

    def sample(self, i: int, j: int) -> np.float32:
        return np.float32(1)


def _damage_and_launch(gpu_backend) -> None:
    """Plant the damage, launch, and pin what the machinery does about it.

    The planting mirrors what this session observed: a generated file whose forward-
    declaration block is missing while its digest sidecar still matches. The launch is
    expected to fail today; the point of the test is what the failure *leaves behind* -- a
    file that is still damaged, so every later launch fails too. The assertions name both
    halves: the launch must eventually succeed, and the file must have been repaired by
    that launch (or by the machinery), not by a hand deletion.
    """
    # Warm the cache first, so the planting edits a file the machinery itself wrote.
    warm = _AnswersOneDamaged()
    warm.draw[gpu_backend]()

    files = generated_files(gpu_backend)
    assert files, "the generated file for this backend was never written"
    from known_failures.damaged_generated_file_driver import strip_prototype_block

    removed = sum(strip_prototype_block(path) for path in files)
    assert removed, "the planted damage did not find a forward-declaration block to strip"

    # The heal: a relaunch after the damage must succeed, WITHOUT any hand deletion, and
    # the file on disk must carry the block again.
    healed = False
    try:
        viewer = _AnswersOneDamaged()
        viewer.draw[gpu_backend]()
        healed = all(marker_present(path) for path in generated_files(gpu_backend))
    except Exception:  # noqa: BLE001 -- still broken is exactly what we are pinning
        healed = False
    assert healed, ("a damaged generated file is never repaired: the digest sidecar still "
                    "matches, so the missing forward-declaration block is never re-added, "
                    "and every later launch fails until the cache is deleted by hand")


def test_a_damaged_generated_file_heals_on_the_next_launch(gpu_backend):
    """Strip the forward-declaration block, keep the digest honest, and launch twice.

    The first launch may fail -- that is the damage showing. The second launch is the
    contract: the machinery rewrites the file from the current render and the launch
    succeeds, with no hand deletion. Today both launches fail and the file is unchanged,
    which is what this test pins until the digest covers the generated text (or a failed
    build triggers a rewrite from the current render).
    """
    _damage_and_launch(gpu_backend)
