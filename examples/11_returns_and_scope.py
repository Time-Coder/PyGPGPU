"""11_returns_and_scope.py -- return values, outer variables, host-side creation.

Three features of a `@kernel` body, all of which keep python's own semantics:

- **Return values.** A kernel runs once per work item, so there is no single C
  return value: every returned position travels as a parameter instead, and the
  first work item to reach a `return` claims the result. You get back the value
  (or tuple) that work item returned, or None when no work item returned at all.
  Different `return` statements may return different shapes.
- **Outer variables.** A variable the body reads from the enclosing scope or the
  module becomes a device parameter, refreshed on every call, so changing it is
  visible immediately. Declaring it `nonlocal`/`global` makes it writable: the
  kernel writes through a pointer and the value is copied back after the launch.
- **Host-side creation.** Statements above the magic loop may build data with
  numpy (`np.zeros`, `np.full`, `np.arange`, `*_like`, ...). They are executed by
  the interpreter, so the array has the shape and dtype numpy gives it, and it is
  passed to the kernel as a parameter. Returning it hands back that same object.

Run:
    python examples/11_returns_and_scope.py
"""
import numpy as np

from pygpgpu import kernel


SCALE = 2.0
seen = 0


@kernel
def first_above(x: np.ndarray, limit: float):
    """Return the first element over `limit` together with its index."""
    for i in range(x.shape[0]):
        if x[i] > limit:
            return x[i], i


@kernel
def scaled(x: np.ndarray, out: np.ndarray):
    """`SCALE` is read from the enclosing module and passed as an argument."""
    for i in range(out.shape[0]):
        out[i] = x[i] * SCALE


@kernel
def accumulate(x: np.ndarray):
    """A declared global scalar is written through a pointer and copied back."""
    global seen
    for i in range(x.shape[0]):
        if i == 0:
            seen = seen + int(x[i])


@kernel
def scaled_by_created(x: np.ndarray, out: np.ndarray):
    """`weights` is created on the host, above the loop, and read on the device."""
    weights = np.full(4, 2.0, dtype=np.float32)
    for i in range(out.shape[0]):
        out[i] = x[i] * weights[i % 4]


@kernel
def doubling_copy(x: np.ndarray):
    """The created array is returned as the very object the host built."""
    result = np.zeros_like(x)
    for i in range(result.shape[0]):
        result[i] = x[i] * 2.0
    return result


def main():
    x = np.array([1.0, 5.0, 12.0, 3.0], dtype=np.float32)

    # 1. a tuple return
    value, index = first_above(x, 10.0)
    print("first above 10     :", float(value), "at", int(index))

    # 2. nothing returns -> None
    print("first above 100    :", first_above(x, 100.0))

    # 3. an outer read variable, and a declared write-back
    out = np.zeros_like(x)
    scaled(x, out)
    print("scaled by SCALE    :", out.tolist())

    # the parameterization is what makes a changed outer value visible at once
    global SCALE
    SCALE = 10.0
    scaled(x, out)
    print("after SCALE = 10   :", out.tolist())

    global seen
    seen = 10
    accumulate(x)
    print("global write-back  :", seen)

    # 4. a host-created array read inside the kernel
    scaled_by_created(x, out)
    expect = x * np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    print("created weights    :", np.allclose(out, expect))

    # 5. a host-created array returned as itself
    result = doubling_copy(x)
    print("doubled copy       :", result.tolist(), "(same object:", result is x, ")")


if __name__ == "__main__":
    main()
