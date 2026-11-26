# Inline kernel blocks: the implemented design

`with inline, kernel(...):` writes a `@kernel` body at its launch site, with no `def`. This
record is the contract -- what the code does, why each piece exists, and the places a change
would break quietly.

## 1. What it is, and what it deliberately is not

A block is **not a second execution model**. The body is the body a decorated kernel would
have: the same magic loop, the same three host regions, the same read/write copy policy, the
same generated file, the same per-kernel cache, the same backends. The translator, the host
regions, the phase split, the argument binding and the launch are all untouched.

What is new is only *where the body lives* and *where its data comes from*:

```python
a   = np.arange(1024, dtype=np.float32)
out = np.zeros(1024, dtype=np.float32)

with inline, kernel(default_device="cuda"):
    for i in range(out.shape[0]):
        out[i] = a[i] * 2.0
```

`a` and `out` are not written anywhere. The names the body reads from the enclosing function
become the rebuilt function's **parameters**, which is exactly what `def k(a, out)` would have
given the same body. That single decision is what lets every downstream stage stay unchanged.

## 2. The mechanism, and why each part is load-bearing

### 2.1 The signal

`inline` is a fixed singleton and must be the **first** `with` item. The block manager
(`_KernelSpec`, returned by `kernel(...)`) raises `InlineBlock` from its `__enter__`; the
singleton's `__exit__` catches it, suppresses it, and launches. Suppressing the exception is
what stops the body from running -- a `with` body only executes when every head item returned
normally.

Four consequences, each load-bearing:

- **`inline` must come first.** `with kernel(...), inline:` enters the manager first, so its
  exception escapes before the catcher is on the stack. That is a real failure, and both
  orders get a message naming the correct spelling (`kernel`'s own `__exit__`, and
  `inline.__exit__` for the `with inline:` alone case).
- **The launch happens in `__exit__`, exactly once.** `InlineBlock.consumed` guards a
  re-raise; without it a body could be launched twice.
- **The result slot is per thread and stacked.** `with inline as result, kernel(...):` binds
  the kernel's return value. The slot cannot live on the singleton (`__enter__` returns it,
  `__exit__` needs it) without two concurrent launches sharing one object, so it is pushed on
  a `threading.local()` stack. `with` statements nest, so the stack is required even
  single-threaded.
- **`kernel(...)` had to become context-manager-shaped.** The overload that carries options
  returns `_KernelSpec`, which is callable (so `@kernel(domain="x")` is unchanged) *and* is a
  real context manager. Declaring `-> object` instead made `ty` reject every block with
  `invalid-context-manager`, which is why the annotation names the real class.

### 2.2 Finding the block

`frame.f_lineno` inside the manager's `__enter__` is always the `with` statement's **first**
line -- verified for one-line heads, multi-line heads, a trailing comment, a backslash
continuation, and a method body. A `with` is compound, so no two can start on the same line:
the line is the identity.

The node is then chosen among the `ast.With` nodes **whose span contains that line**, keeping
the innermost (largest `lineno`) -- an enclosing `with` can span the same line
(`with open(...):\n    with inline, kernel():`). The candidates are filtered to those whose
first item is the `inline` singleton *by identity* and whose last item is a `Call`, so the
match does not depend on the spelling of the name.

### 2.3 Rebuilding the body as a function

The translator and the host-region scan read source through
`translator/source_text.get_dedented_source`, and require `tree.body[0]` to be an
`ast.FunctionDef`. A synthesized text solves that:

- blank lines for `1 .. with_line - 1`, then `def inline_block_<with_line>(<params>):`, then
  the block's own lines dedented by **exactly** the `with` statement's indentation;
- so generated line N is the user's file line N, and the translator's
  `first_line + lineno - 1` mapping becomes the identity -- a translation error points at the
  real line with the real source text under the caret (pinned by
  `test_kernel_inline_kernels_placement.py`).

Dedenting by a constant rather than `textwrap.dedent` matters: a triple-quoted string starting
at column 0 inside the block would defeat a common-prefix computation.

The source is registered in `translator/source_text` (`register_block_source`,
`_BLOCK_SOURCES`, a `WeakKeyDictionary` on the function object). That one registry is read by
all three readers -- `get_dedented_source`, `_source_first_line` (which answers 1 for a
block) and `_source_signature`.

`build_function` then compiles at the user's filename and **rebinds to the module's real
globals dictionary** with `types.FunctionType`. That is deliberate: module globals resolve,
device helpers resolve, `global x = v` lands where it would have, and no per-call refresh is
needed. The compile happens into a throwaway namespace so `inline_block_<line>` never becomes
a module attribute.

### 2.4 Capture: which names become parameters

`captured_names` walks the block body and keeps a name when:

- the block does not bind it itself (`_bound_in_the_block`: assignment targets, `for` targets,
  `with ... as`, imports, comprehension variables, nested `def`/`class` names, `global`/
  `nonlocal`, exception handlers), **and**
- it is in `frame.f_locals`.

The locals test comes **first**: a local shadowing a module global is precisely the value the
body must see, and testing `f_globals` first silently passed the global (pinned by
`test_a_local_shadowing_a_module_global_wins`).

Order is **source order of first read**, sorted by `(lineno, col_offset)` -- `ast.walk` is
breadth-first, so a two-statement body would otherwise interleave its statements. A method's
receiver (`self`/`cls`) is hoisted to the front, because `PyKernel` reads the first
parameter's name to decide what travels separately.

Values are read from `frame.f_locals` at launch time, not at capture time, so a rebinding
between the head and the launch is seen.

### 2.5 The domain is the magic loop's, or a child launch's

An inline block states where its work goes in one of two ways:

1. a **magic loop** in the body (`for i in range(x.shape[0])`), whose bound is the domain;
2. a **child launch** (`child[(n,)](...)`, or a bare `child(...)` when the child has its own
   magic loop), which delegates the division to that kernel -- the block is then the single
   work item that starts it.

A bracket-less child call is sized exactly as a bracket-less host call is: from the child's
own magic loop. Requiring a bracket only in device code made the two spellings of one kernel
disagree about whether a size was needed, so the device path derives it from the child's
extent parameters -- the same values the launch already forwards to the child.

The first is the common case. There is no `domain=` on a block head: the loop says the domain
in the body itself, and a block is written where it is launched, so the array is already in
scope. `domain=` exists for a decorated kernel because that body is written away from the
arrays it runs over. Letting a block carry both would be a second, silent spelling of what the
loop already says.

A gid-style body (`get_global_id`) is still refused by name: it reads a work item whose
extent a block cannot declare. But a child launch is *the* device-side equivalent, and it runs
the same code path a `@kernel` body's `child[(n,)](...)` does -- CUDA's dynamic parallelism.

`blocks_per_grid=` / `threads_per_block=` remain: they do not state the domain, they state how
it is *split into work items*. `blocks_per_grid` alone keeps the bracket's "bare item count"
reading; with `threads_per_block` the pair goes through the same `_parse_work_item` a
`[blocks, threads]` bracket does, so there is one conversion rule.

A device-side launch from a block forwards the child's derived extent parameters
(`int _out_shape0`) the same way a host launch does. That was a real defect: the child's
signature ends with one scalar per extent its magic loop derives, and omitting them is a C
compile error ("too few arguments in function call") far from the line that caused it. Only a
gid-style child derives nothing, which is why the existing tests -- all gid-style -- did not
catch it.

### 2.6 One kernel per site

`_SITES` maps `(filename, with line, block text digest, options digest)` to the `PyKernel` and
the launch shape. The text digest means an edited block is re-translated; the options digest
means a changed `default_device=`/`int=`/`float=`/work-size pair is a different kernel.

Executing one site twice builds one kernel; a test asserts this on the cache length and on
object identity, not on timing.

## 3. Refusals

All are `SyntaxError` at capture time, before translation, because the alternative is a
translator error about syntax the user never wrote:

| block contains | why |
| --- | --- |
| no magic loop, or a gid-style body | the domain has to come from the body itself; `get_global_id` reads a work item whose extent a block cannot declare. This is the one place a block is deliberately narrower than `@kernel` |
| `domain=` on the head | the loop already states the domain; `domain=` is the decorator's spelling |
| a one-line body | no text of its own to rebuild from |
| `return` | a `with` has no value; answer through `with inline as result` instead |
| a top-level `break`/`continue` | they would target the caller's loop, which the block never enters; inside the block's own loop they keep their device meaning |
| `nonlocal` | the rebuilt block is module-level, so there is no enclosing scope to write; capture a one-element array |
| `await`/`yield` | this launch is synchronous |
| a `with inline` inside a `@kernel` body | that body is device code, so it has no `with` and no second launch. Both backends refuse; before this was named, the interpreter ran it as python and answered a different number while a device backend refused |

## 4. Module map

| module | owns |
| --- | --- |
| `inline.py` | the `inline` singleton, its `__exit__`, the per-thread result slots |
| `spec.py` | `_KernelSpec`: the `with`-item head, the decorator half, the option refusals |
| `site.py` | finding the `ast.With`, capture, the capture-time refusals (gid body included), the generated text |
| `rebuild.py` | the generated `def` text and the rebuilt function object |
| `launch.py` | the per-site cache, the `PyKernel` build, the launch call |

## 5. What a change could break quietly

- **The generated text must keep line N == file line N.** Any reflow (re-indenting the body,
  dropping the leading blank lines) shifts every reported error line and breaks the caret
  column, which `_syntax_error` reconstructs from the dedent it assumes.
- **The registry is the only source of truth.** A new reader of a kernel's source must consult
  `block_source`; a direct `inspect.getsource(raw_func)` returns the block's *statements*.
- **The capture list is the parameter list.** A name captured but not read, or read but not
  captured, changes the device signature; capture order is the launch's argument order.
- **The locals-before-globals test is the shadowing rule.** Reversing it is silent.
- **`_SITES` grows for the life of the process**, the same lifetime a `@kernel` cache entry
  has. A site dropped by the program keeps its entry; the key digest is what stops an edited
  file from being served stale text.
- **`inline` is compared by identity.** It is a singleton because `_find_with` and the tests
  check `value is inline`; a factory would break the match.
- **A block inside a `@kernel` body is refused in two places, on purpose.** The interpreter
  backend reaches the `with` while running the body, so the capture refuses it
  (`_refuse_a_block_inside_a_kernel_body`, which reads `host_runtime.get_host_frame`); a device
  backend never runs the body, so `translator/statements.py` refuses the `with` itself. Both
  messages name the launch site, and the pair is what keeps one spelling from meaning two
  things. Removing either check brings back the silent divergence: the interpreter computed
  and the device refused.
