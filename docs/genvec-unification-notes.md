# The genVec to pygf base-class unification: before and after

`pygpgpu/vec_types/genVec.py` (392 lines) was a second, parallel implementation of the
vector base pygf already carries. Every vector class now inherits pygf's `genVec`/`genType`,
the width bases (`genVec2/3/4/8/16`) are thin subclasses of pygf's, and the file is deleted.
This is the review record: what each type answers, before versus after.

The comparison specs are `tests/test_genvec_{arithmetic,swizzles,layout_and_construction,
numpy_view}.py` (510 checks, golden values over every dtype family and width).
pygf's side of the change is release 0.6.0; its own suite (`pygf/tests/test_pygf.py`) pins
the shared behaviors standalone. After this record was written, the 8/16-wide families moved
beside the only backend that declares them (`pygpgpu/opencl/driver/vec_types/`) and kernels
refuse the width everywhere -- see `tests/test_kernel_wide_vector_refusal.py`.

## 1. Layout: byte-identical, for all 64 classes

`np.dtype(cls).itemsize`, `ctypes.sizeof(cls)` and the field list of every class are
unchanged -- the three-way layout split (pygf's 12 shared, pygpgpu's 24 own, 14 CUDA/HIP
driver, 18 OpenCL driver) survives untouched. The generated dump is `docs/vec_type_after_table.txt`
at `docs/vec_type_after_table.txt` (regenerate with the script at the bottom of this file if needed); the spec's
`test_numpy_dtype_matches_fields` and `test_registry_classes_keep_their_layout` pin the
invariants per class. The identity checks:

- `pygf.float2 is CUInfo.vec_types["float2"]` is `True` (and HIP's, and OpenCL's).
- `pygf.float3 is CUInfo.vec_types["float3"]` is `False` -- the compact CUDA/HIP `float3`
  and the OpenCL-padded `float3` (a trailing `_` field, 16 bytes) remain the driver classes.
- Every entry of every registry is a `pygf.genVec` subclass.

## 2. What did NOT change

- The `_fields_` of every class, hence every device ABI layout, structured dtype, and the
  `_vec_registry()` / `CLInfo.vec_types` / `CUInfo.vec_types` tables.
- The generated `.pyi` stubs: regenerated with `write_clmath_pyi()` and byte-identical.
- pygf's own vector surface: `float2/3/4`, `int2/3/4`, `uint2/3/4`, `double2/3/4` answer
  everything they answered before, with three additions (below).
- Kernel-side behavior: the translator reads `_fields_` and emits device swizzles from the
  registered names; nothing in code generation moved.

## 3. Swizzle behavior, before -> after

| expression | before (pygpgpu base) | after (unified) |
| --- | --- | --- |
| `char2(1,2).s0` | worked (swizzle_map) | works (field read) |
| `char4(1,2,3,4).xyz` | `ModuleNotFoundError` (no `char3` in `vec_types`) | `AttributeError` -- the narrow families have no host vec3 (their vec3s are driver device classes) |
| `float8(...).s01` / `.xw` / `.rgb` | `ModuleNotFoundError` (no `float2`/`float3` module locally) | works -- `pygf.float2` / `pygf.float3` |
| `float16(...).sA0`, `.saf` | `ModuleNotFoundError` | works -- `pygf.float2(10, 0)` |
| `float8(...).s01234567` | worked (`float8` local) | works |
| `char16(...).sAF`, `.sfa` | worked | works |
| `char16(...).a` | `10` (hex wins over rgba) | `10` -- preserved |
| `char16(...).rgba` | `(0, 1, 11, 10)` (hex shadows b/a) | same |
| `char4(1,2,3,4).s01 = (3,4)` | worked | works |
| `char4(...).s01 = 3` (scalar) | `ValueError` (scalar refused) | works -- broadcasts (pygf's rule) |
| `float2(...).s0` (pygf class) | n/a | refused -- the numeric gate stays off for pygf's own surface |
| `char2(...).s0x`, `.sxy` | worked (strip-`s` then per-char lookup) | `AttributeError` -- a numeric name is `s` + numeric-charset chars only |
| `char2(...).s0101` (len 4) | worked (`char4`) | works |
| `float8(...).s01234` (len 5) | refused | refused |
| `char4(...).xx = 5` (duplicate) | `AttributeError: no setter` | same message |
| `float2(...).rg = 7` (pygf class) | fields untouched, python shadow attrs set (the bug) | fields written: `(7.0, 7.0)` |

## 4. Construction, before -> after

| expression | before | after |
| --- | --- | --- |
| `char2(1.9, 2.1)` | `TypeError` (floats reached ctypes raw -- the conversion checked `c_int8.__name__`, but the class's real name is `c_byte`) | `(1, 2)` -- C truncation toward zero |
| `char2(-1, 300)` | `(-1, 44)` (ctypes wrap) | same |
| `char2([1, 2])`, `char2((1, 2))` | worked | works (pygf `__init__` flattens sequences now) |
| `char2(float2(1.9, 2.9))` | `TypeError` | `(1, 2)` -- components truncate |
| `char2(1, 2, 3)` | `ValueError: char2.__init__ accepts 2 values, got 3` | `ValueError: invalid arguments for char2()` (pygf's message) |
| `float4(...)` on pygf classes | unchanged | unchanged (plus sequence arguments) |

## 5. Arithmetic, comparison, unary: before -> after

| expression | before | after |
| --- | --- | --- |
| `char2 + char2` | `char2` | `char2` (same answer, pygf's machinery) |
| `char2 + int2` | `ValueError` (the pygf operand broadcast whole-vector into one component) | `int2(11, 22)` -- promotion through pygf's dtype order |
| `float2 + char2` (pygf left) | `ValueError` (same broadcast bug) | `float2(2.0, 3.0)` |
| `char2 / char2` | `TypeError` (floats into `c_byte`) | `pygf.float2(0.5, 1.0)` -- `/` between integer types below `uint` is a true division |
| `ushort2 / ushort2` | n/a (no `%`/`/` at all on the old base) | `ushort2(3, 3)` -- `ushort` ranks above `uint` (it is the `half` storage dtype), division stays in the family and truncates like C |
| `char2 % char2` | `TypeError` (no `__mod__` on the old base) | `char2(1, 2)` |
| `char2 < char2` | `ModuleNotFoundError` (no local `int2`) | `pygf.int2(1, 0)` |
| `float8 < 2` | `int8(1, 0, ...)` (local `int8`) | `int8(1, 0, ...)` -- the comparison int-vector resolves in the operand's own package |
| `char2 == "x"` | `TypeError` | `False` (pygf: a non-homo, non-number operand answers a bool) |
| `~uchar2(1,2)` | `uchar2(254, 253)` | same (pygf gained `__invert__`; a float vector is refused by python's own `~float`) |
| `+char2` | worked | works (pygf gained `__pos__`) |
| `abs(char2(-1, 2))` | `TypeError` (no `__abs__` on the old base) | `char2(1, 2)` |
| `int8 ** 2` | n/a (`is_number` mistook the class for a numpy `np.int8` scalar and broadcast it as an operand) | `int8(9, ...)` -- `is_number` excludes anything carrying `math_form` |

## 6. dtype surface, before -> after

| expression | before | after |
| --- | --- | --- |
| `cls.dtype` (class) | `np.dtype(_fields_)` (plain attribute) | same value, from pygf's descriptor |
| `instance.dtype` | `np.dtype` (the plain attribute answered both) | the ctypes scalar class (`c_byte` for `char2`) -- what pygf's type arithmetic reads |
| `cls.numpy_dtype()` | `np.dtype(_fields_)` | same |

The instance-level change is the one deliberate behavior swap; no pygpgpu code reads an
instance's `.dtype` expecting the numpy form (the registry, layout packing and translator all
read classes), and the spec pins the new shape.

## 7. pygf's own changes (release 0.6.0)

- `genType`: the dtype tables rank the narrow integer ctypes classes (`c_byte`, `c_ubyte`,
  `c_short`, `c_long`, `c_ulong`, `c_longlong`, `c_ulonglong`); `_bin_op_type` resolves the
  result through the operand's own family when pygf carries no module for the dtype; `__pos__`
  and `__invert__` added. The name table is built with `setdefault` because `c_int is c_long`
  on Windows -- a literal dict made `'long'` answer for plain `int` vectors.
- `genVec`: numeric swizzles behind `_numeric_swizzles` (off for pygf's own classes); swizzle
  writes store through the component accessor (the shadow-attribute fix) and accept sequences
  and scalar broadcasts; `__init__` flattens sequence arguments; `_vec_result_type` resolves
  swizzle results in the operand's family; comparisons answer the int vector of the operand's
  package; `is_number` does not mistake a container class named like a numpy scalar.
- `helper`: `generate_getter_swizzles`/`generate_setter_swizzles` take a `lengths` parameter
  (the numeric sets are enumerable only for the lengths they are asked for), `from_import`
  takes a `package` (family resolution beside the operand's own modules), and the swizzle
  setter declarations accept `Sequence[float]`.
- Version 0.6.0; its suite pins each fix standalone.

## 8. Regenerating the class dump

```bash
python - <<'EOF'
import io, ctypes
import pygpgpu
from pygpgpu.opencl.driver import CLInfo
from pygpgpu.cuda.driver import CUInfo
from pygpgpu.hip.driver import HIPInfo

rows, seen = [], set()
for source, label in ((CLInfo.vec_types, "opencl"), (CUInfo.vec_types, "cuda"),
                      (HIPInfo.vec_types, "hip")):
    for name, cls in sorted(source.items()):
        if not isinstance(cls, type) or not hasattr(cls, "numpy_dtype"):
            continue
        key = (name, cls.__module__)
        if key in seen:
            continue
        seen.add(key)
        d = cls.numpy_dtype()
        fields = ", ".join(f"{fn}:{fd}" for fn, fd in d.fields.items())
        rows.append((label, name, cls.__module__, ctypes.sizeof(cls), d.itemsize, fields))

with io.open("vec_type_after_table.txt", "w", encoding="utf-8", newline="\n") as f:
    for label, name, module, size, dsize, fields in rows:
        f.write(f"{label:8} {name:10} sizeof={size:<4} dtype={dsize:<4} {module}\n")
        f.write(f"{'':40} fields: {fields}\n")
EOF
```
