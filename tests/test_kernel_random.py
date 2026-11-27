"""Random numbers in a kernel body: the spellings, the stream, and the refusals.

A kernel body draws the way it would on the host -- `random.random()`,
`np.random.uniform(0.0, 1.0)` -- and the launch owns the seed and the work item, so the
body never passes either. That is the contract `taichi.random` offers, and it is what a
model function needs: a helper that draws takes no `tid` and no `seed`.

What the tests here pin:

* **the spellings.** The draws the standard library and numpy publish are recognized by
  the *object* the callee resolves to, so `from random import random` and `import random`
  agree, and a user's own `def random(...)` stays theirs.
* **the stream.** Two work items are two streams, a work item's k-th draw is the same
  number whatever order the work items ran in, and a pinned seed replays the sequence.
* **the refusals.** A draw answers one number, so `np.random.random(3)` and a `size=`
  argument cannot be represented; and `np.random.seed(0)` is the launch's business, not
  a work item's.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from pygpgpu import get_global_id, kernel
from pygpgpu.kernel import random as random_support


@kernel
def uniform_draw(out: np.ndarray):
    i = get_global_id(0)
    out[i] = random.random()


@kernel
def ranged_draw(out: np.ndarray):
    i = get_global_id(0)
    out[i] = random.uniform(2.0, 4.0)


@kernel
def normal_draw(out: np.ndarray):
    i = get_global_id(0)
    out[i] = random.gauss(10.0, 1.0)


@kernel
def integer_draw(out: np.ndarray):
    i = get_global_id(0)
    out[i] = float(random.randint(1, 6))


@kernel
def numpy_draw(out: np.ndarray):
    i = get_global_id(0)
    out[i] = np.random.random()


@kernel
def numpy_ranged(out: np.ndarray):
    i = get_global_id(0)
    out[i] = np.random.uniform(0.0, 1.0)


@kernel
def two_draws(out: np.ndarray):
    """Two draws in one body, so the second has to follow the first."""
    i = get_global_id(0)
    out[i] = random.random() * 1000.0 + random.random()


def draw_one(kernel_object, backend: str, count: int = 8) -> np.ndarray:
    out = np.zeros(count, dtype=np.float32)
    kernel_object[backend][(count,)](out)
    return out


# ---------------------------------------------------------------------------
# The spellings
# ---------------------------------------------------------------------------


def test_a_local_name_shadowing_the_module_is_not_a_draw(gpu_backend):
    """A body that binds `random` itself owns the name: `random.random()` is then an
    attribute of a float, which python would refuse too."""
    @kernel
    def shadowed(out: np.ndarray):
        for i in range(out.shape[0]):
            random = 3.0
            out[i] = random.random()

    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(SyntaxError) as refused:
        shadowed[gpu_backend](out)
    assert "not a python function" in str(refused.value)


def test_the_standard_library_spellings_are_recognized():
    """The catalogue holds the standard library's own callables, not a copy of them."""
    assert random_support.random_spelling(random.random) is not None
    assert random_support.random_spelling(random.uniform) is not None
    assert random_support.random_spelling(random.gauss) is not None
    assert random_support.random_spelling(random.randint) is not None
    assert random_support.random_spelling(random.randrange) is not None


def test_the_numpy_spellings_are_recognized():
    assert random_support.random_spelling(np.random.random) is not None
    assert random_support.random_spelling(np.random.uniform) is not None
    assert random_support.random_spelling(np.random.normal) is not None
    assert random_support.random_spelling(np.random.randint) is not None


def test_a_users_own_function_is_not_a_draw():
    """Identity, not spelling: a body that defines its own `random` keeps it."""
    def random() -> float:
        return 0.5

    assert random_support.random_spelling(random) is None


def test_an_import_under_another_name_is_still_a_draw(gpu_backend):
    """`from random import random as rnd` and `import random as random_support` both resolve to the
    same callable, and identity is what decides."""
    from random import random as rnd
    import random as rng_module

    @kernel
    def via_from_import(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = rnd()

    @kernel
    def via_module_alias(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = rng_module.random()

    for body in (via_from_import, via_module_alias):
        out = np.zeros(8, dtype=np.float32)
        body[gpu_backend](out)
        assert np.all(out >= 0.0)
        assert np.all(out < 1.0)


def test_an_unhashable_object_is_not_a_draw():
    assert random_support.random_spelling(["not", "a", "function"]) is None


def test_the_two_libraries_keep_their_own_integer_meaning():
    """`random.randint` includes its high end; numpy's `randint` does not."""
    inclusive = random_support.random_spelling(random.randint)
    exclusive = random_support.random_spelling(np.random.randint)
    assert inclusive is not None and exclusive is not None
    assert inclusive[0] != exclusive[0]


# ---------------------------------------------------------------------------
# The values
# ---------------------------------------------------------------------------


def test_the_uniform_draws_are_in_range(gpu_backend):
    values = draw_one(uniform_draw, gpu_backend, 64)
    assert np.all(values >= 0.0)
    assert np.all(values < 1.0)


def test_the_ranged_draws_are_in_range(gpu_backend):
    values = draw_one(ranged_draw, gpu_backend, 64)
    assert np.all(values >= 2.0)
    assert np.all(values < 4.0)


def test_the_normal_draws_have_their_mean_and_spread(gpu_backend):
    values = draw_one(normal_draw, gpu_backend, 4096)
    assert float(values.mean()) == pytest.approx(10.0, abs=0.2)
    assert float(values.std()) == pytest.approx(1.0, abs=0.2)


def test_the_integer_draws_are_integers_in_their_range(gpu_backend):
    values = draw_one(integer_draw, gpu_backend, 256)
    assert np.all(values == np.floor(values))
    assert np.all(values >= 1.0)
    assert np.all(values <= 6.0)


def test_a_negative_range_draws_too(gpu_backend):
    """The bounds may be negative, and each library keeps its own meaning of the high
    end: `random.randint` includes it, numpy's `randint` does not."""
    @kernel
    def inclusive(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = float(random.randint(-5, 5))

    @kernel
    def exclusive(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = float(np.random.randint(-5, 5))

    inclusive_values = np.zeros(512, dtype=np.float32)
    inclusive[gpu_backend](inclusive_values)
    assert np.all(inclusive_values >= -5.0)
    assert np.all(inclusive_values <= 5.0)

    exclusive_values = np.zeros(512, dtype=np.float32)
    exclusive[gpu_backend](exclusive_values)
    assert np.all(exclusive_values >= -5.0)
    assert np.all(exclusive_values <= 4.0)


def test_the_numpy_spellings_draw_too(gpu_backend):
    values = draw_one(numpy_draw, gpu_backend, 64)
    assert np.all(values >= 0.0)
    assert np.all(values < 1.0)
    ranged = draw_one(numpy_ranged, gpu_backend, 64)
    assert np.all(ranged >= 0.0)
    assert np.all(ranged < 1.0)


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------


def test_two_work_items_are_two_streams(gpu_backend):
    values = draw_one(uniform_draw, gpu_backend, 32)
    assert len(set(values.tolist())) == 32


def test_the_host_seed_replays_the_sequence(gpu_backend):
    """The seed is the host generator's, read on the first kernel draw of the process."""
    state = random.getstate()
    try:
        random.seed(20240101)
        random_support._reset_generator_seeds()
        first = draw_one(uniform_draw, gpu_backend, 16)
        random.seed(20240101)
        random_support._reset_generator_seeds()
        second = draw_one(uniform_draw, gpu_backend, 16)
        assert np.array_equal(first, second)
    finally:
        random.setstate(state)
        random_support._reset_generator_seeds()


def test_a_different_host_seed_gives_a_different_sequence(gpu_backend):
    state = random.getstate()
    try:
        random.seed(1)
        random_support._reset_generator_seeds()
        first = draw_one(uniform_draw, gpu_backend, 16)
        random.seed(2)
        random_support._reset_generator_seeds()
        second = draw_one(uniform_draw, gpu_backend, 16)
        assert not np.array_equal(first, second)
    finally:
        random.setstate(state)
        random_support._reset_generator_seeds()


def test_the_numpy_generator_seeds_numpy_draws(gpu_backend):
    state = np.random.get_state()
    try:
        np.random.seed(1234)
        random_support._reset_generator_seeds()
        first = draw_one(numpy_draw, gpu_backend, 16)
        np.random.seed(1234)
        random_support._reset_generator_seeds()
        second = draw_one(numpy_draw, gpu_backend, 16)
        assert np.array_equal(first, second)
    finally:
        np.random.set_state(state)
        random_support._reset_generator_seeds()


def test_the_second_draw_follows_the_first(gpu_backend):
    """Two draws in one body: the second is a different number, so the state advanced."""
    values = draw_one(two_draws, gpu_backend, 32)
    assert np.all(values != np.floor(values))


def test_the_passes_of_one_body_draw_different_numbers(gpu_backend):
    """A body split into passes launches one kernel per pass, and each pass draws from
    its own stream: two passes sharing one would answer the same numbers."""
    @kernel
    def two_passes(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = random.random()
        middle = np.zeros(out.shape[0], dtype=np.float32)
        for i in range(out.shape[0]):
            middle[i] = random.random()
        for i in range(out.shape[0]):
            out[i] = out[i] + middle[i]

    out = np.zeros(16, dtype=np.float32)
    two_passes[gpu_backend](out)
    # Two uniform draws, so each element is below 2 -- unless both passes drew the
    # same number, which would put the element above 1.
    assert np.all(out < 2.0)
    assert np.all(out > 0.0)


def test_a_helper_draws_from_its_callers_stream(gpu_backend):
    """A helper that draws takes no seed and no work item: it uses the caller's state."""
    @kernel
    def via_helper(out: np.ndarray):
        i = get_global_id(0)
        out[i] = helper_draw() + helper_draw()

    out = np.zeros(16, dtype=np.float32)
    via_helper[gpu_backend][(16,)](out)
    assert np.all(out >= 0.0)
    assert np.all(out < 2.0)


def helper_draw() -> float:
    """A device helper that draws: the state is its caller's, so it names neither."""
    return random.random()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def _refusal(kernel_object, backend: str) -> str:
    out = np.zeros(4, dtype=np.float32)
    try:
        kernel_object[backend](out)
    except SyntaxError as error:
        return str(error)
    raise AssertionError("the launch was expected to be refused")


def test_a_draw_cannot_build_an_array(gpu_backend):
    @kernel
    def wants_an_array(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = float(np.random.random(3)[0])

    message = _refusal(wants_an_array, gpu_backend)
    assert "one number per call" in message


def test_a_size_argument_is_refused(gpu_backend):
    @kernel
    def wants_a_size(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = float(np.random.random(size=3)[0])

    message = _refusal(wants_a_size, gpu_backend)
    assert "one number per call" in message


def test_seeding_is_not_a_device_operation(gpu_backend):
    @kernel
    def wants_to_seed(out: np.ndarray):
        for i in range(out.shape[0]):
            np.random.seed(0)
            out[i] = 0.0

    message = _refusal(wants_to_seed, gpu_backend)
    assert "no device meaning" in message
    assert "np.random.seed" in message


def test_an_unsupported_draw_names_the_supported_ones(gpu_backend):
    @kernel
    def wants_a_choice(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = float(random.choice([1, 2, 3]))

    message = _refusal(wants_a_choice, gpu_backend)
    assert "random.choice" in message
    assert "random.random" in message


def test_a_keyword_argument_is_refused(gpu_backend):
    @kernel
    def wants_a_keyword(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = np.random.uniform(0.0, high=1.0)

    message = _refusal(wants_a_keyword, gpu_backend)
    assert "positional" in message


# ---------------------------------------------------------------------------
# The generated code
# ---------------------------------------------------------------------------


def _generated_source(backend: str) -> str:
    """The source the last launch generated, read from the cache."""
    import glob
    import os

    folder = {"cuda": ("__cucache__", ".cu"), "opencl": ("__clcache__", ".cl"),
              "hip": ("__hipcache__", ".hip")}[backend]
    generated = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       folder[0], "pykernels", f"*{folder[1]}"))
    assert generated, "the generated source was not found"
    text = ""
    for path in generated:
        with open(path, "r", encoding="utf-8") as file:
            text += file.read()
    return text


def test_the_generated_source_includes_the_random_header(gpu_backend):
    draw_one(uniform_draw, gpu_backend, 4)
    source = _generated_source(gpu_backend)
    assert "include/random.h" in source
    assert "random_init" in source


def test_the_generated_source_declares_the_launch_seed(gpu_backend):
    draw_one(uniform_draw, gpu_backend, 4)
    source = _generated_source(gpu_backend)
    assert "_random_seed" in source
