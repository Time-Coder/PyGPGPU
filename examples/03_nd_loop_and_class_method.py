"""03_nd_loop_and_class_method.py -- N-dimensional iteration + class methods.

- `np.ndindex(shape)` / `np.ndenumerate(arr)`: the 2D/3D "magic" loops, one
  element per work-item (outermost loop only; for 1D use
  `range(arr.shape[0])` or `enumerate(arr)`).
- `@kernel` also decorates instance methods: `self.xxx` attributes referenced
  in the body are collected automatically and passed as extra device
  arguments, read from the instance at call time. Static methods work with
  either decorator order.
- Helpers may be reached through a name: a static method of any class
  (`Model.double`, `Other.scale`) or another method of the same class
  (`self.clamp`), whose own `self.xxx` reads become kernel arguments too.

Run:
    python examples/03_nd_loop_and_class_method.py
"""
import numpy as np

from pygpgpu import kernel


@kernel
def matrix_add(m: np.ndarray, out: np.ndarray, k: float):
    """Elementwise 2D add of a scalar: (y, x) maps one-to-one to a work-item."""
    for y, x in np.ndindex(out.shape):
        out[y, x] = m[y, x] + k


@kernel
def square_each(m: np.ndarray, out: np.ndarray):
    """np.ndenumerate gives both index and value; each element writes its own slot (naturally parallel)."""
    for (y, x), v in np.ndenumerate(m):
        out[y, x] = v * v


@kernel
def row_sums(m: np.ndarray, out: np.ndarray):
    """Reduction example: the outer range gives one work-item per row, the
    inner range accumulates sequentially within the row."""
    for row in range(out.shape[0]):
        s = 0
        for x in range(m.shape[1]):
            s += m[row, x]
        out[row] = s


class Model:
    # An @kernel-decorated instance method: self.weights / self.bias become
    # device parameters automatically.

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    @kernel
    def apply(self, x: np.ndarray, out: np.ndarray):
        """self.weights / self.bias are collected and passed as device args."""
        for i in range(out.shape[0]):
            out[i] = x[i] * self.weights[i] + self.bias


class Activations:
    @staticmethod
    def double(v):
        """A static method of another class: no receiver, so it is a plain helper."""
        return v * 2.0


class Layer:
    """Helpers reached through a name, including another method of the same class."""

    def __init__(self, scale):
        self.scale = scale

    def clamp01(self, v):
        """An instance method used as a helper: self.scale becomes a kernel argument."""
        return min(max(v, 0.0), self.scale)

    @kernel
    def run(self, x: np.ndarray, out: np.ndarray):
        """Calls self.clamp01 (same class) and Activations.double (another class, static)."""
        for i in range(out.shape[0]):
            out[i] = self.clamp01(Activations.double(x[i]))


@kernel
def static_double(x: np.ndarray, out: np.ndarray):
    """staticmethod can be decorated too (in either decorator order)."""
    for i in range(out.shape[0]):
        out[i] = x[i] * 2


def main():
    # 2D ndindex
    m = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = np.zeros((3, 4), dtype=np.float32)
    matrix_add(m, out, 10.0)
    assert np.allclose(out, m + 10.0)
    print("matrix_add (3x4, 2D) OK")

    # ndenumerate elementwise (parallel-safe: each element writes its own slot)
    sq = np.zeros((3, 4), dtype=np.float32)
    square_each(m, sq)
    assert np.allclose(sq, m * m)
    print("square_each (ndenumerate) OK")

    # Per-row reduction (one work-item per row)
    rowsum = np.zeros(3, dtype=np.float32)
    row_sums(m, rowsum)
    assert np.allclose(rowsum, m.sum(axis=1)), (rowsum, m.sum(axis=1))
    print("row_sums (row reduction) OK:", rowsum.tolist())

    # Instance method: switching backends via brackets still works
    model = Model(np.arange(4, dtype=np.float32), 0.5)
    x = np.ones(4, dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)
    model.apply(x, y)
    assert np.allclose(y, np.arange(4, dtype=np.float32) + 0.5), y
    print("Model.apply (class method) OK:", y.tolist())

    # Static method
    z = np.zeros(4, dtype=np.float32)
    static_double(x, z)
    assert np.allclose(z, x * 2)
    print("static_double OK")

    # Helpers reached through a name: self.clamp01 (same class) calls
    # Activations.double (a static method of another class); self.scale of
    # clamp01 is uploaded as a kernel argument like any other attribute.
    layer = Layer(np.float32(1.5))
    x2 = np.array([-1.0, 0.5, 1.0, 4.0], dtype=np.float32)
    y2 = np.zeros(4, dtype=np.float32)
    layer.run(x2, y2)
    expected = np.minimum(np.maximum(x2 * 2.0, 0.0), 1.5)
    assert np.allclose(y2, expected), (y2, expected)
    print("Layer.run (self method + other-class static) OK:", y2.tolist())


if __name__ == "__main__":
    main()