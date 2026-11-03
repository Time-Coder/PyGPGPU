# kernel_import example

One module name, three backends. This directory is an ordinary python package: it holds
`vecops.cl`, `vecops.cu` and `vecops.hip` -- the same module written once per backend -- and
`kernel_import` merges them, so a single import publishes one set of python objects.

```python
from pygpgpu import kernel_import

kernel_import.install()
from kernel_import import vecops        # this package; `vecops` is the merged module

vecops.add_offset(a, b, out, offset, n)          # default device, like a @kernel call
vecops.add_offset["cuda"](a, b, out, offset, n)  # a bracket names the backend
vecops.add_offset["opencl:1"][(blocks, threads)](...)   # device + work size, either order
```

## What the sources declare

| Kernel | vecops.cl (opencl) | vecops.cu (cuda) | vecops.hip (hip) |
|---|---|---|---|
| `add_offset(a, b, out, offset, n)` | yes | yes | yes |
| `in_circle(points, results, center, radius, n)` | yes | yes | yes |
| `shift(out, factor, n)` | `float` | `double` | `float` |
| `negate(out, n)` | yes | -- | -- |

`Pt` is declared identically by all three files, so the merged module publishes one
`ctypes.Structure` for it and any backend accepts a value built from that class.

## What that means

* **One parameter list, several implementations.** `add_offset` is declared with the same
  parameters in all three files, so the merged module has one `__call__` for it: a bare call
  runs the default device, and a bracket runs the backend it names. `in_circle` is the same
  shape with a struct argument and a struct array.
* **Different parameter lists are overloads.** `shift` takes `float` in `.cl`/`.hip` and
  `double` in `.cu`, so the stub declares two `@overload`s. `shift["cuda"]` reaches the double
  form, `shift["opencl"]` the float one, and a bare call walks the `cuda -> hip -> opencl`
  probe and takes the first declaration the argument types fit -- a `float32` buffer skips
  cuda's double-only declaration.
* **A name one backend lacks raises at the bracket.** `negate` exists only in `vecops.cl`, so
  `negate["cuda"]` raises `KeyError` before any argument is looked at.

The generated stub is `vecops.pyi`, written next to the first source whenever the sources are
newer than it (see `../15_kernel_import.py` for the runnable version of everything above).
