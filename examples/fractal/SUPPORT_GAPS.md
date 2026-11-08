# Spellings PyGPGPU does not accept

Collected while building the perturbation-theory renderer in
`examples/fractal/Mandelbrot_viewer.py`. Every entry was verified on this machine (RTX 4070
Laptop; CUDA 12, ROCm 7.2 gfx1036, OpenCL) by running a kernel that uses the spelling.

## Fixed

| # | Spelling | Behaviour before | Now |
|---|---|---|---|
| 1 | any **host-region** statement that only python can express: `import json`, a `dict`, a string, an arbitrary-precision `mpf`, a `self.refresh()` that touches one | `SyntaxError: unsupported statement: Import` / `'center' is a mpf, which cannot become a structure member yet` | runs on the host, **once**, and the value it binds travels to the kernel as a parameter (see `tests/test_kernel_host_regions.py`) |
| 2 | `2.0 * z`, `z * 2.0`, `z + 2.0`, `z - 2.0`, `2.0 - z`, `z / 2.0`, `2.0 / z`, `z ** 2.0`, `2.0 ** z` -- a **float literal** beside a complex value | `CompileError: identifier "c128_scale_c128" is undefined` (and `_add_real_c128`, `_rsub_c128`, `_rdiv_c128`, `_rpow_c128`, `_pow_real_c128`, `_scale_div_c128`) | all 28 spellings compile and match numpy (see the `FLOAT_SCALAR_FORMS` test in `tests/test_kernel_complex.py`) |
| 3 | `src = a`, `src = self.data` then `src[i]` (a local alias of an array) | `IndexError: list index out of range` | a real pointer assignment -- `float* src = a;` -- with the extents resolved to the array the local points at. Chains (`two = one`) and every domain spelling (`range(src.shape[0])`, `enumerate(src)`, `np.ndindex(src.shape[...])`) follow it (see `tests/test_kernel_array_views.py`) |
| 4 | `part = a[lo:hi]` then `part[i]` (a slice) | `IndexError: list index out of range` | a pointer at an offset (`float* part = a + lo;`) with the slice's **own length**, so `part.shape[0]` is the slice's length and a store through it reaches the array. `a[2:][i]` folds the offset into the subscript |
| 5 | `a[-1]` (a negative index) | read `a + (-1)` -- **before the buffer**, answering `0.0` where python answers the last element; `a[-1] = 99.0` wrote outside the array | a constant negative index folds to `extent - n`; a computed one is passed through (see "Still open"). Per axis in a tuple subscript, and through an alias or a slice |
| 6 | `print(f"slot {i}")` | `SyntaxError: unsupported expression: JoinedStr` | an f-string is the print pieces python already has: its literal runs are text and its `{expression}` fields are values. A conversion (`!r`) or a format spec (`:>4`) is refused by name, because python's formatter has no device counterpart |
| 7 | `for i, w in enumerate([1.0, 2.0, ...])` (a literal list as the domain) | `SyntaxError: unsupported expression: List` | a literal list is a compile-time table: the dispatch bound is its length and each element is a constant in a `switch`, so no buffer is built and no argument is passed. Works inline or bound to a name above the loop (see `tests/test_kernel_helper_tuples.py`) |
| 8 | `lo, hi = bounds(row, column)` where `bounds` returns `row, column` | `SyntaxError: cannot tell the type of the value this helper answers` (in the loop) / `unknown name 'hi'` (above it) | the element types are read from the helper's own returns -- a cast, a constant, or arithmetic over the helper's parameters. Inside the loop the pair travels by address; above it the host calls the helper and each name is a scalar parameter |

## Still open

| # | Spelling | Today | Why it matters |
|---|---|---|---|
| 1 | `a[k]` with a **computed** negative `k` | passed through as the offset it is, so it reads before the buffer | the device has never bounds-checked, so this is the same class of mistake as `a[100]` on a short array. A *literal* `a[-1]` is folded, which is the spelling that actually appears in code; wrapping every access in a sign test would evaluate the expression twice and cost every kernel a select |
| 2 | `a[::-1]` (a reversed slice) | `SyntaxError: a reversed slice is not a device pointer ... use a[n - 1 - i]` | a work item cannot walk a buffer backwards with one, so the refusal is right -- listed because the spelling is common enough to be worth a clear message |
| 3 | `a[1:][1:]` (a slice of a slice) | `SyntaxError: a slice of a slice is not supported` | take one slice into a local and index it |
| 4 | `break` inside the magic loop | `SyntaxError: break cannot stop a magic loop` | correct as it stands (work items run in parallel); listed only for completeness |
| 5 | a nested python list iterated level by level (`for row in grid:`) | `for x in arr` always yields one element, so a row has no slice type | pass the list whole and index it as `grid[i, j]` |

## Checked and working -- no action needed

- `math.pi` and other module constants above the loop
- `"abc".upper()`, `len(...)`, `min(3, 5)` above the loop
- `lo, hi = 1, 3` (a tuple of literals) above the loop
- `pick = 2 if flag else 5` above the loop
- `0 < i < 3` (a chained comparison)
- `abs(round(x))` inside the loop
- `self.a[i] = v; self.a[i] += 1.0` (augmented store into an element)
- a `while` loop inside the work item
- a `@staticmethod` helper called from the body
- `np.float64(2.5)` above the loop
- `7 // 2`, `7 % 3` above the loop
- `self.pick()` returning one value, called above the loop
- `total = 0; ...; total = total + x[i]` with `return total` inside the loop (one shared
  location for the whole launch)
- `return 1, 2` from a kernel with a magic loop (the pair is the call's result)
