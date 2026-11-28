# Automatic differentiation: design and plan

**Status:** M0–M4 landed and pinned by tests; M5 landed (the deferred launch spellings and the static per-phase multi-pass adjoint) (`tests/test_ad_forward.py`,
`tests/test_ad_reverse.py`, `tests/test_ad_refusals.py`, `tests/test_ad_rules_complete.py`,
`tests/test_ad_tape.py`). The code lives in `pygpgpu/ad/`; the machinework this plan decided
is described in `AGENTS.md` (the "Automatic differentiation" section) and in the modules'
own docstrings. This file keeps the design rationale and the open questions; M5 (multi-pass
and async) and M6 (deferred) are still open.

Sections are labelled so a reader can tell what is fixed from what is still an argument:

* **Decision** - settled; implement it this way unless something is found that invalidates the
  reason.
* **Open** - a real question with more than one defensible answer; listed under
  [Open questions](#open-questions) with the recommendation that currently looks right.
* **Non-goal** - named so it is not re-argued per file.

---

## 1. The situation this design starts from

Three facts about this repository decide almost everything below.

1. **A kernel body's source is readable Python, and there is already an AST translator.**
   `@kernel` requires a real `.py` file because `inspect.getsource` is how the body is read
   (`get_dedented_source`), and `FunctionTranslator` walks that tree and emits device C. A
   differentiable kernel can therefore be *differentiated at the source level*, before
   translation, and the result is an ordinary Python function that the existing translator
   already knows how to emit.
2. **The translation is per work item.** A magic loop is not a loop on the device; it is one
   work item per element, and the body is emitted once. That means the reverse of a
   straight-line body is *the same parallel shape*: one adjoint work item per forward work
   item, with local values in registers.
3. **The arrays are already the storage.** `pygpgpu.numpy` arrays hold their own device
   residency and carry a family `epoch`. A tape over kernel launches therefore needs to record
   *references and versions*, not copies.

Warp and Taichi get to the same place by a different route, and the route is worth stating
correctly because the first draft of this file got it wrong. **Neither of them records device
operations.** A `@wp.kernel` body is a Python function that Warp's own front end parses into its
IR, and Warp generates the kernel's adjoint from that IR; Taichi compiles a kernel and derives a
`grad` counterpart from its IR the same way. Their tapes (`wp.Tape`, `ti.ad.Tape`) only
book-keep *which launches ran, in what order*, so the adjoints can be replayed backwards. So this
is not "tape versus no tape" - all three have a launch-level tape. The two real differences are
**which layer the adjoint is generated from**, and **whether the forward and the reverse are one
kernel or two**.

| | Warp | Taichi | PyGPGPU (this plan) |
| --- | --- | --- | --- |
| Kernel body | Python, parsed into Warp IR | Taichi DSL | Python, already parsed for the device translator |
| Adjoint generated from | Warp IR | Taichi IR, after compilation | the Python AST, before translation |
| Launch record | `wp.Tape`: which launch, which arrays | `ti.ad.Tape`: which launch | `Tape`: which launch, `(array, epoch)` |
| Forward values for the reverse | read from the arrays in a separate adjoint kernel | read from `needs_grad` fields as they happen to be | fused: registers, recomputed; multi-kernel: the caller's arrays, epoch-checked |
| Forward + reverse | two kernels | two kernels | one kernel (M2); two per pass for a multi-pass body (M5) |
| Stale forward state | caller's responsibility | documented warning | refused at `backward`, naming the array |
| Reference for checking | `tape.check_grad()` (finite differences) | none built in | the `python` backend, plus finite differences |

### 1.1 Where this plan is genuinely different

1. **Fused forward-and-reverse in one kernel (M2).** Neither framework does this: both emit a
   second kernel that re-reads the forward values from memory, which is why the intermediates have
   to be materialised in an array or field for the backward to find. One work item doing
   forward-then-reverse in registers needs none of that, and it is the only way the reverse walk
   can see a per-work-item local at all. The cost is arithmetic - the forward is computed twice -
   against two launches and the memory traffic of a second pass. For an elementwise kernel that
   second pass is the dominant cost, so the trade is usually favourable; for an expensive body
   over arrays that stay in cache it is not, which is why the `vjp`-across-kernels form (M4) stays.
2. **The epoch check.** Warp and Taichi both rely on "the arrays still hold the forward values" as
   an unwritten contract. Taichi documents the danger; Warp leaves it to the caller. `gnp`'s
   `epoch` makes it a one-integer check, so a violation is a refusal instead of a plausible wrong
   number.
3. **The interpreter is an independent oracle.** Warp's CPU path is compiled C++; Taichi's
   architectures are all compiled. The `python` backend *executes the adjoint body as written*, so
   it is a second implementation of the same derivative, which is what makes `check_grad` a real
   test rather than two compilations agreeing with each other.

### 1.2 What they have that this plan does not, yet

1. **A user-registered adjoint.** Warp's `@wp.func_grad` and Taichi's `ti.ad.grad_for` let a user
   differentiate a black-box helper analytically. Without one, every helper is either
   differentiable by the walk or refused, which is too strict for anything that iterates
   (a Newton step, an integrator). This is a gap in M3, not an optional extra - see
   [Open question 6](#8-open-questions).
2. **Vector and matrix rules.** Warp carries full derivatives for `wp.vec3`/`mat33`/quaternions.
   The plan's rule table is written for scalars; `dot`, `cross`, swizzles and matrix products need
   their own forward and reverse rules, and a vector argument must not silently fall back to a
   component-wise guess where the operation is not component-wise (`dot` is the obvious one).
3. **Parameter and optimizer ergonomics.** `wp.optim.Adam` treats "an array with a gradient" as a
   first-class training concept. The plan stops at `.grad`; whether that is enough depends on
   whether PyGPGPU wants to be used for training loops or only for differentiable simulation.

**Decision:** differentiate the Python AST, and emit the adjoint through the existing
`FunctionTranslator`. No backend gains a code path of its own, and no device-side tape exists in
the design.

---

## 2. Interface

Four layers, preceded by the vocabulary they are written in. The first two are the ones a user
meets; `Tape` is the general mechanism; the power tools are the vocabulary the other three are
explained with, which is also why they are defined in full rather than mentioned.

### 2.1 Array layer: `requires_grad` / `.grad`

```python
import pygpgpu.numpy as gnp

x = gnp.ones(1024, np.float32)
x.requires_grad = True
y = x * 2.0          # .grad is allocated lazily on first backward, same shape/dtype/device
```

**Decision:** `requires_grad` is a plain attribute, not a constructor keyword. The gnp
constructor surface mirrors numpy's signatures (`np.zeros`, `np.array`, `*_like`), and a
`requires_grad=` keyword that numpy does not have would break that mirror. `.grad` is a gnp
array on the same device, so a gradient does not come home unless something reads it.

**Decision:** `.grad` is not inherited by views. A view passed to a kernel tracks because the
view itself says so; propagating a flag through `x[1:]` would make "which object did the caller
actually pass" ambiguous in a library whose binding layer already has to reason about object
identity.

### 2.2 What the words mean

A kernel writes its result into a parameter, and that result is usually an **array**. One fact
follows, and the API is explicit about it rather than hiding it:

> `d(out)/d(x)` for two arrays of length `n` is an `n x n` Jacobian - not an array.

So "the gradient of a kernel" is undefined until the result is reduced to **one number**, and
this API never performs that reduction on the caller's behalf: a default of "sum the output" is
wrong for every loss that is not a sum, and a plausible-but-wrong gradient is the one failure
mode worth designing against. Each word below names one piece of that reduction.

The one spelling that needs no reduction is a `return`, so **that is the default**. A body that
`return`s one number has already done the reduction itself, in python, where the caller can read
it; taking that number is not the API inventing `sum(out)`, it is the API reading an answer the
body wrote down. So `loss` and `out` both default to the string `"return"`, and naming a
parameter is only needed when the body answers through one instead - the reduction-by-atomics
shape, or a result another kernel wrote. `"return"` is a reserved name, so no parameter can
shadow it.

| word | means | appears in |
| --- | --- | --- |
| `wrt` | "with respect to": the parameters the derivative is taken of; defaults to the arrays flagged `requires_grad` | `k.grad`, `k.vjp`, `k.jvp` |
| `loss` | the single number the gradient is of; defaults to `"return"` | `k.grad`, `k.value_and_grad` |
| `out` | the parameter carrying the kernel's array result; defaults to `"return"` | `k.vjp`, `k.jvp` |
| `tangent` | an input direction `dx` (forward mode) | `k.jvp` |
| `cotangent` | an output weight `g` (reverse mode) | `pullback(g)` |

### 2.3 Differentiating a kernel: `k.grad(...)` / `k.value_and_grad(...)`

The transform is spelled on the kernel object, not as a function taking one, because the object
is already where a launch is configured: `k[(4, 256)](args)` names a bracket and then calls, and
`k.grad()(args)` names a derivative and then calls. One shape to learn, and an editor that
completes `k.` shows all of it.

```python
@kernel(domain="x")
def squared_error(x, target):
    total = 0.0                          # a carried scalar: one location for the launch
    for i in range(x.shape[0]):
        total += (x[i] - target[i]) ** 2
    return total                         # the reduction is the body's own answer

x.requires_grad = True                   # this is what `wrt` defaults to
gx, = squared_error.grad()(x, target)
# `gx` is a gnp array with x's shape and dtype, and gx[i] == d loss / d x[i].
```

Read it as *"the gradient form of `squared_error`, of the number it returns"*. Nothing is named
at all - not even `wrt`, which is inferred - because the body already said which quantity it is.
`wrt=("x", "y")` asks for both and answers `(gx, gy)` in that order.

The other spelling names a parameter, and it is still needed when the body answers through one:

```python
@kernel(domain="x")
def squared_error(x, target, loss):      # loss is a shape-(1,) array: the sum of squares
    for i in range(x.shape[0]):
        atomic_add(loss, 0, (x[i] - target[i]) ** 2)

loss = gnp.zeros(1, np.float32)          # the forward pass accumulates into it
gx, = squared_error.grad(loss="loss")(x, target, loss)
```

The gradient of an `n`-element input against a scalar loss is an `n`-element vector, which is why
one number is all it takes for `gx` to exist: every element's own partial derivative, nothing
off-diagonal. That is the difference between `grad` and `vjp`, and it is the whole reason the two
are separate forms.

**Decision:** `wrt` defaults to **every argument whose `requires_grad` is set**, and an explicit
`wrt=` overrides it. The array flag is the declaration; requiring both the flag and the name list
would be two statements of one fact, and the inference is checkable at the call (the arrays are
there), which is exactly where a mistake is cheap to report. An empty set is refused - a
derivative with respect to nothing is a mistake, not an empty answer.

**Decision:** the configuring call takes **keywords only**. `k.grad(x, out)` is refused with a
message naming the second call, because the two-call shape is the whole point: the first call
names the derivative, the second is the kernel's own call and keeps the kernel's own signature.

**Decision:** `loss` defaults to `"return"`, and that is the preferred spelling. A body that
`return`s one number has already performed the reduction, so the API takes it rather than
inventing one. A body that returns nothing, whose return sites disagree in arity or width, or
whose returned value is not launch-wide - the `_return_is_shared` rule: a carried scalar or a
kernel argument, so every work item computes the same number - is refused with a message naming
both spellings. `loss="name"` stays because it is the only form that can accumulate across work
items through an atomic (the shape above), and because a body may already answer through a
parameter the caller holds.

**Decision:** either spelling must name **one number**; nothing is inferred from argument order,
from which parameter the body happens to write, or from the output's length.

**Decision:** `k.grad` returns the **gradients only**, always a tuple ordered by `wrt`;
`k.value_and_grad` is the same call that also gives back the loss the forward computed, as
`(loss_value, gradients)`. They are two forms because two different things are wanted: one call
should not hand back a pair that has to be decomposed when only the gradients were asked for.

**Decision:** this compiles to **one kernel**, not two. Each work item recomputes its forward
values in registers and then walks its statement list backwards accumulating gradients. Local
scalars are never saved, because a work item's local is a register and re-running one statement
is cheaper than any tape entry.

Two consequences are load-bearing and should be stated in the docstring rather than discovered:

* Data-dependent `if`/`else` works in fused mode without any recording, because the predicate
  is per work item and deterministic, so the adjoint re-evaluates it and walks the branch that
  was taken.
* `while` works too, at a cost: the iteration count is a private local, and the reverse walk
  replays the body once per iteration. That is quadratic in the trip count and still needs no
  memory, which is the right trade for the shapes kernels usually run.

### 2.4 A result that is not a scalar: `k.vjp(...)` / `k.jvp(...)`

When the answer is an array, the operation that is always defined is the **pullback**: hand it a
weight for every element of the output, and it answers a gradient of the input's shape.

```python
@kernel(domain="out")
def saxpy(a, x, y, out):
    for i in range(out.shape[0]):
        out[i] = a * x[i] + y[i]
    return out                           # an array result is legal: one object, named by all

out.requires_grad = False                # the result: left unflagged, so not in `wrt`
result, pullback = saxpy.vjp()(2.0, x, y, out)
gx, gy = pullback(gnp.ones_like(result))    # d sum(out)/dx and d sum(out)/dy
gx, gy = pullback(w)                        # d sum(w*out)/dx and dy, for any weight w
```

`result` is the caller's own `out` object, returned for symmetry with a functional call. What
matters is `pullback(g)`: it answers `J^T g`, the gradient of `sum(g * out)`, as a tuple ordered
by `wrt`. `k.grad` is a thin wrapper over this - seed 1.0 onto the one scalar it was given, and
read the gradient - which is why `grad` needs a scalar and `vjp` does not.

**Decision:** `out` defaults to `"return"` for the same reason `loss` does: a body that returns
the array it wrote has said which result it is. `out="name"` is for a body that returns nothing,
or a result a *different* pass wrote - which is `Tape`'s case (2.5) at the multi-kernel scale.

Forward mode asks the transposed question, and its answer has the **output's** shape rather than
a gradient's:

```python
dout = saxpy.jvp(tangent=dx)(2.0, x, y, out)   # with x.requires_grad set
# dout[i] == d out[i] / d t, for x(t) = x + t * dx
```

**Decision:** `tangent` is bound by the configuring call, not appended to the kernel's call. The
call's signature then stays exactly the kernel's own, which is what lets a type checker keep
checking `k.jvp(...)(args)` the way it checks `k(args)` - `_P` already carries the parameter list
(PEP 612), and a trailing extra argument would step outside it.

For a **function that is several kernels** there is no kernel object to hang these on, so the
same three transforms exist as functions in `pygpgpu.ad` (`pygpgpu.ad.grad(fn, ...)`); they are
implemented once and both spellings reach it. A sequence of launches with host code between them
is `Tape` (2.5), which is a different subject: it records what ran rather than transforming
something known ahead of time.

**Decision:** `jvp` is built first (see [Milestones](#5-milestones)) even though `grad` is what
users ask for, because forward mode is a directional derivative of the same loop: no reverse
walk, no accumulation shape, no scatter. It proves the AST-transform skeleton on something that
cannot be subtly wrong about ordering.

**Decision:** `check_grad` is a first-class deliverable, not a test helper. The `python`
backend runs the adjoint body serially, so it *is* a reference implementation of the gradient,
which means AD correctness is testable in full on a machine with no GPU - the same property the
rest of the package already relies on.

### 2.5 Several kernels: `Tape`

```python
from pygpgpu import Tape

with Tape() as tape:
    out = gnp.zeros_like(x)
    saxpy(2.0, x, y, out)
    dot_kernel(out, out, loss)      # loss is a shape-(1,) gnp array

tape.backward(loss)                 # seed 1.0 into the one-element array
print(x.grad)                       # still on the device
```

**Decision:** `Tape` records **kernel launches only**, never device operations. The host python
between launches is already the "host region" the decorator runs, so the launch is exactly the
unit that has a device meaning, and it is the unit whose adjoint is a kernel.

**Decision:** the tape records each launch's kernel, its argument bindings and the
`(array, epoch)` of every tracked array. Between the forward and backward pass the arrays keep
their forward contents in place, so the tape is memory-free; what it must check is that those
contents are still the ones the forward used, which is the `epoch` comparison in 4.5.

**Decision:** `tape.backward` takes the **array itself**, not a `"return"` or a parameter name.
The `"return"` default exists because a kernel's call has no value to point at, only a signature;
a launch that already ran does have one, so the caller passes the object it holds. That is the
same reason `Tape` needs no `wrt`: `requires_grad` already marked the inputs.


---

## 3. Where the machinery lives

| Path | Subject | Must not contain |
| --- | --- | --- |
| `pygpgpu/ad/__init__.py` | public surface: `grad`, `value_and_grad`, `vjp`, `jvp`, `check_grad`, `Tape` | transformation logic |
| `pygpgpu/ad/forward.py` | forward-mode transform (tangent propagation over the AST) | reverse-mode rules |
| `pygpgpu/ad/reverse.py` | reverse-mode transform: statement ordering and accumulation | per-op rules |
| `pygpgpu/ad/rules/` | one register entry per differentiable name, keyed by the **object** | traversal |
| `pygpgpu/ad/tape.py` | launch recording, seeding, epoch validation | code generation |
| `pygpgpu/ad/ast_support.py` | helpers that build adjoint expressions from forward ones | any one op's rule |
| `pygpgpu/ad/errors.py` | the named refusals, positioned like `translator/errors.py` | rules |

The four properties on the kernel object are not a fifth module: `PyKernel.grad` and its three
siblings are thin, lazily-importing accessors that build a form from `pygpgpu.ad` (see 4.8), so
`decorator/` gains four short properties and no transformation code.

**Decision:** the transform sits between the source and the translator, as a new stage, not
inside `FunctionTranslator`. The translator's job stays "one body in, device C out"; the AD
stage's job is "one body in, another body out". Keeping them separate is what lets the adjoint
body be inspected, cached, printed and run on the interpreter.

**Decision:** the one seam this needs in existing code is in `translator/source_text.py`: the
translator has to accept a **parsed AST** as well as text, because the Python 3.7 floor has no
`ast.unparse`, so an adjoint cannot be turned back into source text to be re-parsed. The
existing cache already hands the translator a deep copy of a parsed tree, so this is a narrow
change: a variant of `_parse_kernel_source` whose "source" is a tree, and a key for the cache
that names the transform that produced it.

**Decision:** the AD rule set is part of the kernel identity. A VJP change moves the gradient
kernel's body while the user's `.py` has not moved, so the rules' version goes into `_c_name`'s
hash and into the cache key, beside `type_map` and the helper-closure digest. Without it a rule
fix is served the binary compiled from the old rule.

---

## 4. Decisions

Each entry is written as the decision, then the reason, because the reason is what a later
reader needs in order to know whether it still holds.

### 4.1 Rules are registered by object identity, and completeness is enforced

**Decision:** the VJP table is keyed by the builtin's *object*, exactly like
`builtin_registry.builtin_name_of`. A call is differentiable because the callee resolves to an
object that has a rule, never because its spelling matches.

**Decision:** `tests/test_builtins.py`'s existing agreement check becomes a four-way check:

> Every builtin has either a VJP rule or a `NON_DIFFERENTIABLE` entry with a reason.

The sentence this prevents is: "a new builtin was added and it silently has a zero gradient."
That is the worst failure mode available here, because the forward answer is right, the
expression type-checks, and only the number that comes out of backward is wrong.

### 4.2 The privacy proof is reused, not rewritten

A subscript that is provably private to the work item makes a gather: the adjoint is
`g_x[i] += g_out[i] * d`. Anything else makes a scatter, which needs an atomic add.

**Decision:** reuse the proof `_augassign_is_race_free` already makes for the atomic conversion
of `arr[i] op= value`. One proof, two consumers. A second copy of that analysis would drift, and
the drift would show up as a silently missing contribution - the same class of bug the atomic
conversion itself exists to prevent.

**Decision:** a reduction is not a special case. `total += x[i]` on a carried scalar is already
an atomic add on the forward side, so its adjoint is `g_x[i] += g_total`, which is a plain
atomic add on the scalar's gradient. Nothing new is needed to express it.

### 4.3 Refuse by name, with a position

**Decision:** anything the device cannot differentiate is refused at translation time with a
`SyntaxError` carrying the line and the replacement spelling - never a silent zero gradient.
The list starts with: an in-place write into an array whose `requires_grad` is set, `atomic_*`,
`print`, bitwise operators, `mul24`/`mad24`, and `raise`/`assert`.

This is the same philosophy as `break` in a magic loop: what the device genuinely cannot
express is said out loud, and what it can express less precisely is not.

### 4.4 Gradients follow the same type mapping

**Decision:** a gradient buffer is declared through the same `type_map` / `float=` mapping as
the forward array, so `float=np.float64` gives a double-precision gradient. A gradient that
silently narrowed to 32 bits would be invisible in every sklearn-style shape check.

**Decision:** complex is out of scope for v1 and refused by name. Reverse mode over `c64`/`c128`
needs the Wirtinger convention settled (which of the two partials a "gradient" is), and that is a
documented decision, not a branch added in passing. The refusal points at the two-real-arrays
spelling.

### 4.5 `epoch` is the tape's version number

**Decision:** a recorded launch stores `(array, family.epoch)`, and `backward` refuses if any
epoch has moved since the forward ran, naming the array.

Warp's and Taichi's tapes have to invent this versioning; `pygpgpu.numpy` already has it, and
`epoch == current_epoch` is precisely "the host bytes still describe what the device wrote".
Using anything weaker would let a legitimate-looking backward pass read a buffer the caller
overwrote in between, which is a wrong-number bug with no crash.

### 4.6 The adjoint kernel is an ordinary kernel

**Decision:** no new binding, transfer or launch code. The adjoint kernel reads `g_out`, writes
`g_x`, so `KernelFlags`' analysis gives it the right copy policy for free; `c_name` hashing, the
`__clcache__` machinery and the multi-pass `submit`/`async_call` chain all apply unchanged.

A multi-pass forward becomes "forward passes in order, then adjoint passes in reverse order",
which is the same done-callback chain with more stages - no thread ever waits on the GPU, and
the ordering is the only new rule.

### 4.7 Non-goals

* **No operator overloading.** The body is an AST; there is no evaluation point to overload, and
  overloading would leak into device code.
* **No device-side tape.** It would need a memory story per backend to buy "saving instead of
  recomputing", and recomputing is cheaper for a per-work-item body.
* **No global switch** (`config.autodiff = True` or similar). Differentiability belongs to the
  array (`requires_grad`) and to the call site (`k.grad` / `Tape`), not to a process.
* **No host-side numpy AD in v1.** Differentiating `out.sum()` between two launches means
  writing a numpy AD; the pattern instead is a loss kernel, or a `Tape` over one launch.
* **No `@kernel(differentiable=True)` marker.** Reading `k.grad` does not change the kernel: it
  builds a form for this call, and the plain `k(args)` call is untouched.

### 4.8 The transform hangs on the kernel object, and the object imports it lazily

**Decision:** `PyKernel` carries `grad`, `value_and_grad`, `vjp` and `jvp` as properties that
return a *form* - a small object holding the configuration, callable with the kernel's own
parameters. It is the same two-call shape the bracket already uses (`k[(4, 256)](args)`), so the
object has one way of saying "configure, then call". Three costs are paid deliberately:

1. **The import stays lazy.** `import pygpgpu` must not pull the AST transform in, so each
   property imports `pygpgpu.ad` on first access. Import cost is measured in this package (the
   benchmark note about the pre-3.9 AST shim), and a module that only a differentiable kernel
   needs does not belong on that path.
2. **The bound form carries the four properties too.** `@kernel` on a method returns
   `_BoundKernelMethod` from `__get__`, so `obj.trace.grad(...)` has to work or a differentiable
   method cannot be differentiated from an instance - and a method body is a common shape (the
   ray tracer's `trace`). The properties live on the base both wrappers share, never copied.
3. **The name overlaps an array's `.grad`, and that is accepted.** `x.grad` is a buffer;
   `k.grad(...)` is a transform that returns a callable. Different objects, different call
   shapes. `k.grad_of` or `k.derivative` would disambiguate but read worse for the call people
   type most, and the docstring's first line settles it.

---

## 5. Milestones

Ordered so each one is testable on its own and each one's failure mode is narrow. `[ ]` is not
started, `[x]` is landed and pinned by tests.

- [x] **M0 - this document.** Kept current; a milestone's decisions leave here when they land.
- [x] **M1 - `jvp`, fused.** The AST-transform skeleton: a tangent variable per tracked name,
  one pass over the statement list, per-op forward rules, `out` defaulting to `"return"`, emitted
  as a kernel by the existing translator. Exit criterion: `jvp` matches a hand-written derivative
  on int32/float32/float64 for elementwise bodies, on the `python` backend and every available
  device backend. *(Landed: `ad/forward.py`, `ad/operator_rules.py`, `tests/test_ad_forward.py`.)*
- [x] **M2 - `k.grad`, fused reverse.** The four kernel-object properties (2.3, 2.4, 4.8),
  statement list walked backwards, accumulation into registers, adjoined stores written
  elementwise (gather only), `wrt` inferred from `requires_grad` with an explicit override,
  `loss`/`out` defaulting to `"return"`, `k.value_and_grad`, `check_grad`. Exit criterion:
  `check_grad` passes for a single-store body, a two-statement chain, a body with an `if`, and a
  body with a `while`; a body that returns a launch-wide scalar differentiates with no `loss=`
  named, and one that returns nothing or a per-work-item value is refused naming both spellings;
  `wrt=()` and a positional configuring call are refused; a body with a scatter index is refused
  with a position and a message naming the atomic-scatter milestone. *(Landed: `ad/reverse.py`
  and its mixins, `ad/forms.py` + `ad/build.py` + `ad/forward_form.py`, `ad/check_grad.py`,
  `tests/test_ad_reverse.py`, `tests/test_ad_refusals.py`.)*
- [x] **M3 - the rules table is complete.** Every entry of `SHARED_MATH_BUILTINS`,
  `SHARED_MATH_TEMPLATES` and the python builtins is either ruled or listed in
  `NON_DIFFERENTIABLE`; the four-way test in 4.1 is in place; the vector rules (`dot`, `cross`,
  swizzles, matrix products) are either ruled or refused by name; and a user can register an
  adjoint for their own device helper, so a black-box body is not a dead end. Exit criterion:
  adding a builtin without a rule fails a test, and a helper with a registered adjoint
  differentiates against `check_grad`. *(Landed: `ad/rules/`, `ad/grad_for.py`,
  `tests/test_ad_rules_complete.py`.)*
- [x] **M4 - `Tape` across kernels.** Launch recording, seeding, epoch validation, adjoint
  launches in reverse order, scatter adjoints through atomics. Exit criterion: a two-kernel
  chain (forward + loss) matches `check_grad`; a mutated-after-forward array is refused naming
  the array; a reduction's adjoint adds exactly once per element. *(Landed: `ad/tape.py`,
  `kernel/tape_hook.py`, `tests/test_ad_tape.py`.)*
- [x] **M5 - multi-pass and async.** *Deferred launches*: `k.grad().submit(...)` /
  `async_call(...)` launch the single fused adjoint kernel without waiting, so the future
  resolving IS "settled after the last adjoint pass"; the gradient buffers travel as the
  launch arguments (read them through `.grad` after `.result()`). `value_and_grad`, `vjp` and
  a scalar-returning `jvp` refuse the deferred spellings with the reason. *Multi-pass*:
  `ad/multi_pass.py` builds one fused adjoint kernel per pass and launches them last-first --
  forward passes in order (one plain forward), then adjoint passes in reverse, the
  intermediates' cotangent buffers connecting them (each pass's adjoint recomputes its own
  forward, so no stored trace is needed). The supported shape is narrow and everything
  outside it is refused with the reason: consecutive magic loops (no host gaps), one cotangent
  flowing array per pass (the loss accumulation is the seed sink, not a flow), intermediates
  as caller-passed parameters, and the carried-scalar `loss="return"` refused because the
  per-phase replays would double it. Note discovered while testing: a body-created
  intermediate used across passes is refused by the phase machinery's FORWARD itself (a phase
  kernel does not receive a prefix-created array), before the AD layer runs.
- [ ] **M6 - deferred.** Anything that needs a per-work-item stored trace: cross-work-item
  control flow whose predicate depends on another work item's write, and a `while` whose body
  cannot be replayed. Expected to stay unimplemented; recorded here so nobody re-derives why.

---

## 6. Risks and known traps

These are the places where the design is most likely to produce a plausible wrong number, so
each one gets a test when its milestone lands rather than a paragraph when it is noticed.

1. **The same array passed twice.** `k(a, a)` is bound once by `seen_arrays`, but it has *two*
   gradient contributions per work item and they must both accumulate. Deduplicating by object
   here - the correct rule for buffers - is the wrong rule for gradients.
2. **A non-contiguous view.** A strided gnp view keeps a private contiguous copy
   (`_member_bytes`' `offset == -1` path) and scatters back through the view's own layout
   (`_absorb`). Its gradient has to scatter back the same way; writing gradient bytes
   contiguously shifts every row.
3. **An update in place.** A body that *reads and writes* the same argument array
   (`x[i] = x[i] * 2`) has an adjoint that needs the pre-state, which the fused form no longer has
   (the store overwrote it). A body that only writes (`out[i] = ...`, the saxpy shape) needs
   nothing: `out`'s own gradient is zero because nothing read it. v1 refuses the first case by name
   (4.3) instead of pretending the post-state is the input.
4. **`write_only` qualifiers.** The copy policy ignores them for uploads on purpose; the AD pass
   must not read them as "this buffer's forward value is uninteresting", because the adjoint may
   still need it.
5. **A zero-valued slot.** Truthiness must never decide whether a gradient slot exists:
   `bool(np.array([0]))` is False, and the host-region code already carries a comment about
   exactly this trap.
6. **Aliasing of the seed.** `tape.backward(loss)` where `loss` is also a forward input means the
   seed is read from a buffer the backward writes into. Either a copy of the seed or an explicit
   refusal; decide when M4 lands.
7. **A launch-wide value with a per-work-item shadow.** `total` in the 2.3 example is carried: its
   final value is one number for the launch, while every work item's reverse walk owns its own
   shadow of it. The rule that keeps them apart is that the *value* is read from its slot (so a
   downstream `total * total` is the same constant in every work item) and only the shadow of a
   private local differs per item. The other half is inherited: a body that writes that slot
   non-atomically (`total = total + x[i]`) is already refused on the forward side by
   `_non_atomic_writes`, and the AD pass must let that refusal through rather than differentiate
   the race into a plausible wrong number.

---

## 7. Test plan

* `tests/test_ad_forward.py` - `jvp` against analytic derivatives, every float width, both
  backends available on the machine.
* `tests/test_ad_reverse.py` - fused `grad` against `check_grad`, including the `if`/`while`
  shapes, the `"return"` default for `loss` and each of its three refusals, and every named
  refusal.
* `tests/test_ad_rules_complete.py` - the four-way agreement test from 4.1.
* `tests/test_ad_tape.py` - multi-kernel chains, epoch refusal, the `k(a, a)` accumulation, the
  reduction adjoint.
* `tests/test_ad_refusals.py` - every entry of the named-refusal list, checked at translation
  time with the position and the message.

The `python` backend carries most of the weight: it runs the adjoint body as written, so a test
that passes there and fails on a device is a translation bug, and one that fails in both is an
AD bug. Write every test to run over every backend (`conftest.py`'s parametrization) and read a
failure accordingly.

---

## 8. Open questions

Each of these changes the public API, so each needs an answer before M2. The recommendation is
first; the alternatives are recorded because they were argued for.

1. **Does `.grad` accumulate or overwrite?**
   Recommendation: accumulate (PyTorch semantics), with an explicit `zero_grad()` / `Tape.reset()`.
   Accumulating is what makes "sum several losses" work without a special API, and the reset is
   easy to see in the source. The counter-argument is that a forgotten reset is a silent wrong
   answer, which is why the reset would be explicit on the tape rather than implicit per launch.
2. **`wrt` inferred, by name, or by position?**
   Recommendation: inferred from `requires_grad`, overridden by an explicit name list. The array
   flag is already the declaration, and inferring it at the call is the one place the answer is
   cheap to check. Positions are not offered: a kernel's parameters have names, and `wrt=1` says
   less than `wrt="x"`.
3. **Where does the scalar loss come from - a named parameter, a `return`, or both?**
   Recommendation: both, defaulting to `"return"` and inferring nothing. A `return` is the
   spelling that needs no second statement of intent: the body already performed the reduction,
   so the API reads an answer instead of inventing `sum(out)`. A named parameter carries the
   reduction-by-atomics shape, which a `return` cannot express because the sum exists only after
   every work item has contributed. Argument order and "whichever parameter the body writes" are
   not signals.
4. **Does `Tape` record host-region statements?**
   Recommendation: no in M4. A host region that computes a loss (`loss = out.sum()`) is a
   numpy-AD problem, and accepting it half-way is worse than refusing it by name with the
   "write a loss kernel" message.
5. **Should `vjp` hand back the forward output at all?**
   Recommendation: yes, for symmetry with a functional call, even though an out-parameter kernel
   means the caller already holds that object. The alternative is `pullback = vjp(...)(...)`,
   which is shorter but makes `vjp` the only entry point that answers one thing instead of the
   (primal, derivative) pair. Small question; decided when M2's `vjp` lands.
6. **Does a user-registered adjoint belong in v1?**
   Recommendation: yes, in M3. Warp's `@wp.func_grad` and Taichi's `ti.ad.grad_for` exist because
   a helper is where an iteration lives, and an iteration is not a place to differentiate through.
   The shape that fits this package is a decorator on the helper that names its adjoint's body -
   `@grad_for(newton_raphson)` - so the adjoint is itself a Python function the translator emits,
   never a host-side closure that the device cannot call. The alternative - refusing any helper
   whose body the walk cannot differentiate - makes the feature unusable in exactly the kernels
   that need it most.
7. **Is `.grad` enough, or is an optimizer part of the scope?**
   Recommendation: `.grad` only in v1, with `Tape.reset()`; an optimizer is a separate package
   decision. Warp ships `wp.optim.Adam` because simulation-as-training is its use case, and
   whether that is PyGPGPU's is a product question, not a design question.
8. **Where does the gradient of a *member* go?** (`self.points`, `obj.field` -- the M5 audit.)
   Answered for v1.5 by refusal, not support: `wrt` inference reads the call's ARGUMENTS, so a
   `requires_grad` flag on a member would be silently ignored -- the worst failure mode -- so
   `_check_tracked_members` refuses a flagged member at the call, and the scan-level guards
   refuse member STORES (a store the write analysis cannot see would silently drop its
   contribution) and member READS the cotangent reaches. The alternative spelling is always
   named: pass the member as its own array argument. An UNFLAGGED member scalar read is a
   constant to the adjoint -- dropping its (unrequested) contribution is correct, and the
   tracked gradients stay right. Supporting members means a gradient write-back through the
   `OuterObjectBinding.written` machinery -- a separate decision.
9. **Outer-scope values?** An outer SCALAR read is a constant: dropped without a dead
   cotangent register (`BodyScan.is_outer_name` decides "outer" = never bound in the body and
   not a parameter). An outer ARRAY read is refused with the pass-it-as-a-parameter message;
   supporting it would mean the form allocating gradients for values the caller did not pass
   at the call, which the `wrt` contract has no place to return through.
