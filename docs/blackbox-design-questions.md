# Design decisions taken from the black-box pass

This file records what the second (black-box) discovery round found and what was decided
for each item. The first four columns are the finding; the last is what the code now does.

## Decided and implemented

### 1. A float stored into a narrow integer array wraps like numpy

```python
out = np.zeros(1, np.uint8)
out[0] = -1.0          # numpy stores 255; C's float->unsigned conversion is UB and gave 0
```

**Decision:** match python/numpy. `_wrap_for_store` casts the value through a signed integer
of the destination's width before the unsigned cast, which makes the conversion defined and
gives numpy's modulo. Pinned by `tests/test_narrow_integer_stores.py`.

### 2. Arithmetic written back into a narrow array wraps like numpy

```python
out = np.zeros(1, np.uint8)
out[0] = np.uint8(255) + 1     # numpy wraps to 0; the device answered 256
```

**Decision:** the **store** wraps to the destination's element width (the chosen scope: the
element's own read width stays `int` per the documented `infer_ctype` contract). Pinned by
the same test module.

### 3. `raise` keeps python's exception type, message and line

```python
raise ValueError("the input must not be negative")
# was: RuntimeError: CUDA_ERROR_LAUNCH_FAILED   now: ValueError("the input must not be negative")
```

**Decision:** the failure travels as **data**. The raising work item writes the id of its
raise site into a launch-scoped slot and sets a stop flag; every other work item leaves at
the next loop checkpoint, and the host rebuilds the exception from the site table. The
message must be a **string literal** -- a computed one is refused at translation time,
because the device has no string type to carry it and losing the text silently would be
worse. A bare `raise ValueError` (no message) is supported. Pinned by
`tests/test_kernel_raise_assert.py`.

### 4. `assert` is the same mechanism

`assert cond, "text"` compiles to the same slot write and the host raises python's
`AssertionError` with that text. Pinned by the same module.

### 5. A numeric format spec in `print(f"{x:.2f}")` becomes a printf conversion

**Decision:** support the numeric subset printf can spell --
`[[fill]align][sign][0][width][.precision][type]` with `type` in `f`/`e`/`g`/`d` -- and keep
refusing what printf genuinely cannot express (a `,` thousands separator, a `%` scale, a
string field, a `!r` conversion), naming the reason. Pinned by
`tests/test_print_format_spec.py`.

### 6. A 0-d array argument

No change: the device's refusal ("too many indices for array: array is 0-dimensional") is
what numpy itself raises for the same expression. The interpreter accepts it only because
it never indexes. Working as intended.

### 7. An int too wide for its mapping

Already handled for an **argument** (a `ValueError` naming the numpy spelling that carries
the width) and for a **literal product** (folded and range-checked). The remaining shape --
a module-level constant inlined into the source -- was a real gap and is now checked too
(see `test_kernel_silent_wrong_answer.py`). A value produced *inside* the body by
accumulation stays out of scope: the width is the `int=` contract, and `@kernel(int=np.int64)`
is the documented escape hatch.
