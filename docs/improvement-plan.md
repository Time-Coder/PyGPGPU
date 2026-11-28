# Improvement plan (2026-11-28 full-tree review)

Provenance: five parallel subject reviews (kernel pipeline, backend layer, the uncommitted
AD stage, tests/tooling, public API/packaging) plus a main-session cross-check. Every defect
listed below was verified by reading the code at the cited lines, and the highest-severity
ones were reproduced before being listed. Machine context: Windows, CUDA (RTX 4070) + OpenCL
devices, Python 3.14.4, numpy 2.4.4. Full suite on that machine: **5780 passed / 602 skipped
/ 0 failed in 16m24s**; all repo gates green (ruff, ty, line_budget, any_budget, dup_defs).
Line numbers are current as of this date.

## Severity used here

- **P0** — blocks committing the AD stage (crash on a supported construct, raw `TypeError`).
- **P1** — device-side wrong numbers or leaked device resources on paths users hit.
- **P2** — performance, error-message quality, UX.

---

## P0 — fix before committing the AD stage

> **Status (2026-11-28, same day): all three P0 items below are fixed and tested** —
> the `transform_if` argument fix plus a four-backend branch test, real Tape support
> for method kernels (the receiver is recorded and prepended, mirroring the forms)
> plus a three-backend test, and the `for`/`while` `else` refusal at three checkpoints
> (forward `transform_stmt`, reverse `walk_stmt`, reverse `build` for the dispatch
> loop) with three refusal tests. The full-suite run after the fixes surfaced one
> pre-existing flaky test — `standard_t`'s +-20 bound sits inside the t(5) tail
> (measured: 66/3000 launches exceed it, max 38) — fixed by widening the bound to
> +-80 (`tests/test_kernel_random_distributions.py`, math in the comment there).

1. **JVP crashes on any `if` in the dispatch.** `transform_block()` takes one argument
   (`pygpgpu/ad/forward.py:81`) but `transform_if` passes two
   (`pygpgpu/ad/forward.py:238-239`): `self.transform_block(list(stmt.body), stmt.lineno)`.
   Every branched body raises a raw
   `TypeError: ... takes 2 positional arguments but 3 were given` on every backend — the
   raw-crash shape the feature's own design promises never to show. `copied(stmt)` nodes
   already carry their own lines, so the fix is dropping the stray argument.
   **Also add a forward-mode branch test** (`tests/test_ad_forward.py` has none; reverse
   covers one at `tests/test_ad_reverse.py:63`).
2. **`Tape` over a method kernel crashes or misbinds.** `pygpgpu/ad/tape.py:161-162` is a
   no-op branch (`if kernel._is_method: pass`) while the adjoint keeps the origin's
   signature, so the receiver is never recorded (`pykernel_invoke.py` records `args` only)
   and the backward launch is short one argument — reproduced on OpenCL as
   `TypeError: ... missing 1 required positional argument: '_c_out'`; with other signature
   shapes it misbinds silently. Fix: pass `instance` through `tape_hook.record` and prepend
   it in the backward launch, or refuse method kernels at `record` with the reason.
   **Add a tape-over-method test.**
3. **`for ... else` / `while ... else` are silently dropped in both transforms.**
   `pygpgpu/ad/forward.py:242-250` transforms only `.body`; `pygpgpu/ad/reverse.py:307-323`
   copies `stmt` without walking `orelse`. A value assigned only in the `else` block loses
   its tangent/adjoint silently — a plausible-wrong-number path where everything else in
   the stage refuses. Refuse until handled (break/continue already are, at
   `pygpgpu/ad/reverse.py:207-210`).

## P1 — device correctness and resources

> **Status (2026-11-28, same day): items 4-10 below are fixed.** 4: an error bracket
> around binding and launch on all four backends (`_release_failed_launch` per backend,
> unit-tested for cuda/hip in `tests/test_pooled_buffer_claims.py`). 5-6: Metal's
> `_bound_entries` instance state is gone -- `_bind_args` answers per-launch entries the
> caller carries, and the async poller settles its future on any failure and releases the
> command buffer on every exit path (the `_enqueue` docstring is now true). 7: CUDA
> refuses mixed dtypes exactly as HIP does, and both backends' `out=` writes the
> caller's array through `np.copyto` from a temp seeded with the caller's values
> (`beta` reads C, so `out` is an input too -- caught by the existing
> `test_gemm_honours_alpha_and_beta` while the first cut of the fix zeroed it).
> 9: `NVRTCProgram.ptx()` tries the PTX cache instead of satisfying itself with a cubin,
> `cubin()` falls back to a cached PTX (cuModuleLoadData JITs PTX), and both name the
> failure instead of raising a raw KeyError. 10: the OpenCL binary path names
> `CL_INVALID_BINARY` correctly and no longer turns a ctypes-level failure into a
> `NameError`.

4. **Claimed buffers leak on a mid-launch exception — every device backend, sync and
   async.** Argument binding is a plain loop with no `try/finally`:
   `pygpgpu/cuda/oop/kernel/_kernel_launch.py:40-44`, `pygpgpu/opencl/oop/kernel/__init__.py:208-219`,
   `pygpgpu/metal/oop/kernel/__init__.py:258-261`. A `TypeError` on argument 3 leaves the
   buffers claimed for arguments 1-2 held forever (the pool holds the strong reference; the
   weakref finalizer never runs). Repeated failed launches bleed device memory. Fix: as
   each buffer is claimed, record its serial into the per-call `bound` dict, and on any
   exception walk the partial `bound` and `unuse(serial)` each entry. The stream is already
   released in a `finally` (`pygpgpu/cuda/oop/kernel/__init__.py:121-128`) — this closes the
   other half of the same bracket.
5. **Metal's per-kernel `_bound_entries` races concurrent submits.** `_bind_args` resets
   `self._bound_entries = []` (`pygpgpu/metal/oop/kernel/__init__.py:240`) and
   `take_bound_entries()` consumes only after encode+commit (`launch.py:167`) — two
   interleaved `submit()` calls lose the first launch's entries (missed copy-back, leaked
   claim). Fix: append to a local list returned by `_bind_args` instead of instance state.
6. **Metal's async poller has no error bracket.** `watch()`
   (`pygpgpu/metal/oop/kernel/launch.py:126-142`) reads status, copies back, settles,
   releases — with no `try/except`; any failure leaves the future unsettled forever (an
   `await` hangs) and skips the command-buffer release. The `_enqueue` docstring
   (`launch.py:153-154`) claims an encode-time failure releases the buffer claims — the code
   only releases the command buffer (:164-166). Wrap `watch()` in `try/except` that settles
   the future and releases; make the docstring true.
7. **CUDA `gemm` silently mis-reads mixed-dtype operands.** `dtype = np.result_type(a, b)`
   (`pygpgpu/cuda/libraries/wrappers/_wrappers_blas.py:76-78`) but the operands are never
   cast: float32 × float64 uploads float32 bytes and cuBLAS reads them as doubles — wrong
   numbers, no error. HIP refuses the same input
   (`pygpgpu/hip/libraries/wrappers/_wrappers_blas.py:46-48`). Related: CUDA `out=` handling
   runs `result.astype(dtype)` (:97), so a dtype-mismatched `out` is a copy the caller never
   sees — the caller's `out` is silently not filled. Fix: refuse mixed dtypes exactly as HIP
   does, and make the two backends agree on `out=` semantics (either write through a
   temporary + copy back, or document the return value as authoritative, on both).
8. **Tape epoch validation covers only flagged arrays.** `pygpgpu/ad/tape.py:74-77` records
   `(array, epoch)` only for `requires_grad` arrays; an unflagged input the loss reaches
   (`y` in saxpy) overwritten between forward and backward makes the replay read new bytes —
   a silently wrong gradient. Record epochs for every gnp argument.
9. **`NVRTCProgram.ptx()` can raise a raw KeyError from a healthy cache.**
   `pygpgpu/cuda/oop/nvrtc_program.py:174-181` only tries `_load_cubin`, then indexes
   `self._ptx[device]`; a cache holding the `.cubin` but not the `.ptx` skips `_build` and
   raises. `_load_ptx` (:157-172) is dead code, and `cubin()` rebuilds even when a valid
   `.ptx` is cached. Fix the fallback, delete or wire the dead method.
10. **OpenCL binary-path error reporting.** `_create_with_binary` prints `CL_INVALID_VALUE`
    in the message for the `CL_INVALID_BINARY` branch (`pygpgpu/opencl/oop/program.py:116-117`),
    and `except BaseException: pass` (:103-107) turns a python-level failure into a
    `NameError` at :129 instead of the real cause. Fix the code name; replace the bare pass
    with a re-raise carrying context.

## P2 — performance, UX, noise

> **Status (2026-11-28, same day): all four items below are fixed.** 11: gnp forwards
> numpy's namespace through a PEP 562 module `__getattr__` instead of a star import, the
> decorator's dead `gnp` import is deleted, and asyncio is imported inside the async
> functions (or under TYPE_CHECKING where only an annotation names it) — `import
> pygpgpu` now loads neither `numpy.f2py` nor `asyncio`, measured ~4.4 s → ~2.1 s. 12:
> the probe records why each backend failed and `_resolve_device` raises one error
> naming every runtime's reason plus the `kernel["python"]` escape hatch (pinned in
> `tests/test_probe_order.py`). 13: a range-loop target nothing ever read (whole-body
> scan, `AugAssign` counts as a read) gains neither the per-pass assignment nor the
> hoisted declaration — the nvcc `#550-D` noise is gone at the source. 14: the raw
> stderr traceback before the curated error is removed. `check.py` now checks without
> `--fix`, the `*.pyi.in` package-data glob is gone (the templates are gone), the stale
> README pointer in `cuda/libraries/__init__.py` is inlined, `Tape` is in the top-level
> `__all__`, and the AD plan/AGENTS.md status contradictions are reconciled. Two
> reviewed-and-kept items: `docs/vec_type_after_table.txt` is the genvec notes'
> regeneration target (not a stray dump), and the two `11_` example numbers are
> referenced consistently in both READMEs.

11. **`import pygpgpu` costs ~4.4 s, ~2.9 s of it accidental.** Measured with an import
    hook: `pygpgpu/kernel/decorator/__init__.py:32` imports gnp eagerly;
    `pygpgpu/numpy/__init__.py:48` does `from numpy import *`, which on numpy 2.x triggers
    the module's lazy `__getattr__` and loads `numpy.f2py` -> crackfortran ->
    charset_normalizer -> numpy.testing -> asyncio. With that chain pre-warmed the same
    import measures ~1.5 s. `asyncio` is also imported unconditionally at
    `pygpgpu/kernel/decorator/__init__.py:2` though only the async paths need it. Fix:
    (a) replace the star import with a PEP 562 module `__getattr__` forwarding to numpy on
    demand (keep explicit re-exports for the names gnp defines itself);
    (b) defer the gnp import into the functions that use it — precedent in-tree at
    `pygpgpu/kernel/decorator/_phase_readback.py:70-71`;
    (c) move `asyncio` into the async modules.
12. **No-device UX misleads.** `_probe_backend` swallows every exception with no record
    (`pygpgpu/kernel/decorator/devices.py:212-222`); when all backends find nothing the
    builtin probe falls back to the literal `"opencl"` (:225-240) and the launch fails deep
    with `RuntimeError("no opencl devices found")` (:392) — naming only the last backend.
    Fix: collect per-backend failure reasons and raise one error naming them all plus the
    `k["python"]` escape hatch; the `last_init_error` diagnostics already exist and nothing
    surfaces them.
13. **Generated CUDA declares a dead `int _i`.** One nvcc `warning #550-D` per compile per
    kernel — 8670 warnings in one full suite run, all surfaced to users as `CompileWarning`.
    Delete the dead declaration at the emission site.
14. **Raw traceback on stderr before the curated error.** `pykernel_call.py:120-124` prints
    the untrimmed internal traceback, then raises the trimmed one. Remove it or gate it
    behind a config flag (default off).

---

## Design decisions to revisit

1. **`LaunchPlan` has no declared vocabulary.** `pygpgpu/kernel/decorator/launch_plan.py:16-21`:
   `__getattr__` answers `None` for any unset name and the phases bind ~100 names — a
   misspelled read silently yields `None`, and ty cannot flag it. Declare the names (one
   class-level annotation block) and make `__getattr__` raise for anything outside. Cheapest
   reliability win on the launch path.
2. **Layering inversion: translator imports up into decorator.**
   `pygpgpu/kernel/translator/nested_launch.py:94,246,270` import `_dependency_order`,
   `_positional_argument_names`, `_declares_a_magic_loop` from `..decorator`. Move the
   generic ordering helper down (beside the device-cache subject it orders) so the edge
   points the documented way. Note the two same-named sibling modules
   (`decorator/host_regions.py`, `translator/host_regions.py`) when navigating — the move
   should also reduce the collision's blast radius.
3. **Name-based builtin dispatch inside the AD walk violates the object-keyed rule.**
   `pygpgpu/ad/reverse.py:330-334` matches `spell == "atomic_add"` and
   `spell.startswith("get_")`; `pygpgpu/ad/body_scan.py:191-194` matches
   `startswith("atomic_")`. Consequences: an aliased `atomic_add` used for the legal loss
   accumulation is refused; any user helper named `get_*` bypasses the value-call check.
   Resolve the callee to its object (`ast_support.resolve_callee` is available at both
   sites) and dispatch on that — the rule AGENTS.md already states for the translator.
4. **Receiver-dispatch preamble is four copies already diverging.**
   `pygpgpu/kernel/decorator/__init__.py` `__call__` (:272-293), `submit` (:296-328),
   `async_call` (:331-351) plus `call_proxies.py:58-100` — the next receiver rule must be
   made in four places (one comment has already drifted). Extract one helper.
5. **Two tables own the same 13-entry dtype fact.**
   `pygpgpu/kernel/decorator/slots.py:39-55` `_SCALAR_SLOT_DTYPES` re-spells the numpy half
   of `pygpgpu/kernel/translator/__init__.py:276-308` `RETURN_SLOT_TYPES`. AGENTS.md's own
   "a duplicated table is one constant" rule is met everywhere else here but not here. Let
   one module own `device name -> np dtype` and the other add the C spelling.
6. **`dup_defs.py` cannot see class bodies** — and the tree already has the artifact it
   hunts: `pygpgpu/kernel/translator/__init__.py:264` and `:268` both bind
   `RETURN_CLAIM:str = "_ret_claim"`. Extend the walk one level into `ClassDef.body` with
   the same same-value qualification, delete the dead copy, and collapse that file's doubled
   blank lines.
7. **The Python 3.7 floor is expensive and guarded by nothing.** Its cost: the
   `typing_extensions` dependency, the `tree-sitter-cuda37` fork (upstream needs >=3.10),
   the tree-sitter-metal ABI-13 wheel (built for the 0.20.x line, the last with cp37
   wheels), and the no-`importlib.metadata` version lookup. Its guard: none — ruff targets
   `py312` and ty checks `3.12` while tests run on 3.14, so a `match` statement or a
   3.8+-only stdlib use ships. Also `setuptools>=68` resolves on 3.7 only through the
   68.0/68.1 line (68.2+ requires >=3.8) — the sdist builds there by luck of the resolve,
   not by policy. Recommendation: raise the floor to >=3.9 or >=3.10 and delete the three
   compensations; if 3.7 must stay, add a py37-targeted lint job so the claim is enforced.
8. **Silent process-wide side effects are undocumented in user docs.** `import pygpgpu`
   patches `numpy.array` process-wide (via pygf); `from pygpgpu import *` shadows 5
   builtins (`any, all, pow, min, max`) and 33 numpy names. Both are deliberate; neither is
   in the README. Document both, and put the patch caveat in the module docstring too.
9. **The AD walk's remaining rough edges**: `sign` gets a zero partial while
   `step`/`floor`/`ceil` are refused as piecewise-constant — defensible subgradient
   convention, but undocumented (`rules/scalar.py:275` vs `:356-377`); `forward.py:381-402`
   duplicates callee resolution without the closure-cell walk the reverse path has (reuse
   `ast_support.resolve_callee`); `rules/scalar.py:32` declares `INJECTS:dict`
   unparameterized; `reverse_distribute.py:36` types `Contribution` as a 2-tuple while
   buffer targets are 3-tuples (:219,325) — two named shapes would fix both.

---

## Verification and tooling debt

1. **ty's ignore set is broader than its noise source.** `pyproject.toml:249-263` ignores 13
   diagnostic kinds globally — including `call-non-callable`,
   `too-many-positional-arguments`, `index-out-of-bounds` — to silence ctypes-layer noise
   that `src.exclude` (`pygpgpu/**/driver/**`, :241) already scopes. The translator,
   decorator, AD and vec_types code gets no protection from defect-shaped rule classes.
   Move the ignores into per-path overrides limited to the ctypes/OOP globs; restore the
   three defect-shaped rules globally first.
2. **The gates themselves are untested, and two have bugs.**
   - `any_budget.py:138-147` walks the whole value of an annotation and records every nested
     mention, so one alias counts many times — `constants.py:14` consumes ~5 of the 85
     slots. Count leaf mentions once per annotation.
   - `line_budget.py:56-64` carries 7 allowlist entries whose files now sit at 42-392 lines,
     and the staleness check (:131-135) fires only for missing files — never "this entry can
     be deleted now". Flag at-or-under-ceiling entries for removal, then prune all 7.
   - The warning band holds 87 files, three at 499-500 (`kernel/translator/init.py`,
     `builtin_emit.py`, `opencl/oop/program_parser/__init__.py`) — the pressure is real.
   - The three gates are pure AST functions: give them a test module feeding synthetic files.
3. **`check.py` mutates the tree as part of verifying it.** `check.py:14` runs
   `ruff check --fix`; combined with no CI, lint fixes happen silently and only on the
   author's machine. Check with plain `ruff check`; fix explicitly.
4. **No CI at all — and AGENTS.md contradicts itself about it.** No `.github/` exists
   (verified), while AGENTS.md says both "No CI and no pre-commit hook" and "CI supplies a
   real Apple-Silicon runner". Proposed shape:
   - **Tier 0 (zero hardware, highest value):** one GitHub Actions file, ubuntu + windows:
     `pip install -e ".[dev]"` -> `python check.py` (after 3) -> `python -m pytest tests -q`.
     A GPU-less runner automatically executes the python backend plus every host-side
     structural suite (Metal codegen/launch-path, tables, genvec, probe order, libraries
     loading, config) — the majority of the 6382 collected tests and every currently
     unguarded codegen table. Add a `filterwarnings` policy so new deprecations surface.
   - **Tier 1 (hardware):** windows-latest for CUDA; a self-runner for HIP; a macOS runner
     for Metal, where a VM is useless (`MTLCreateSystemDefaultDevice` returns nil).
5. **Silent-skip degradation has no floor.** `tests/conftest.py:35-36` skips device tests on
   a GPU-less machine (correct per test), but nothing reports that ~101 files' device tests
   vanished. Add a session-level signal: when zero device backends were probed, surface one
   summary line (or fail unless `PYGPGPU_PY_ONLY=1`). Doubles as the tripwire for Tier-1
   runners when the device disappears.
6. **Coverage holes that matter most:**
   - `pygpgpu/utils/work_size.py` heuristics (`detect_work_size`, `local_size_for_global`,
     warp alignment, device-limit clamping) have no unit tests — pure host functions.
   - The driver error tables (`clinfo_sig_*.py`, `clinfo_errors.py`, `check_error`) have no
     direct test — ironically the very tables whose lost hints produced `dup_defs.py`. A
     table-level test (every named ErrorCode resolves, every referenced message exists)
     pins the failure class forward.
   - `examples/` — the documented end-to-end smoke path — is not wired into pytest.
7. **Test smells to clean:** dead `assert_close` in `tests/conftest.py:52-53` (512 direct
   `np.testing` calls); `tests/_helpers/test_*` files named `test_*` without tests;
   `tests/test_kernel_import.py:46-50` installs the import hook and never uninstalls
   (`pygpgpu/kernel_import/hook.py:35-37` has the uninstall); the HIP skip in
   `tests/test_kernel_classmethod_kernels.py:84-88` acknowledges cache-mixing coupling
   instead of isolating the cache.

---

## Packaging, compliance, docs

1. **Vendored pcpp ships without its license.** `pygpgpu/cparser/pcpp/LICENSE` (BSD) is not
   in `package-data` (`pyproject.toml:145-149`) — the 0.2.4 wheel confirms only `.py` files
   ship under `pcpp/`. Add it to package-data and add a NOTICE (or a README third-party
   section) covering pcpp, the three tree-sitter grammars (including the house-built
   `tree-sitter-metal`) and the verbatim SDK headers.
2. **pygf has no runtime version guard.** No `__version__` to check and no upper bound;
   an incompatible pygf surfaces as a bare ImportError/AttributeError at
   `pygpgpu/__init__.py:36` with no hint of the required `pygf>=0.6.0`. Wrap the import and
   name the requirement on failure.
3. **Exception surface is fragmented.** `pygpgpu.exceptions.CompileError` is not exported
   top-level; `LibraryError` is reachable only via `pygpgpu.libraries`. Pick one home users
   can catch, and export the others from it.
4. **Wrapper vocabulary is not backend-portable.** CUDA `to_device(pointer, array)` vs HIP
   `to_device(array)`; the top-level proxy exposes 11 names while `pygpgpu.cuda.libraries`
   exposes 14 (`allocate` works through one path, not the other). Derive the proxy exports
   from the library package's own list (the `kernel_import/tables.py` pattern) and unify or
   rename the two `to_device` shapes.
5. **Doc drift to fix in one pass:** `docs/autodiff-plan.md:3-8` contradicts itself
   ("M5 landed" vs "M5 and M6 still open"; milestone list marks M5 `[x]` at :500); the
   AGENTS.md AD section header says "(M0-M4 landed)" while the section documents M5 content;
   AGENTS.md says both "No CI" and "CI supplies a real Apple-Silicon runner";
   `pyproject.toml:149` declares `**/*.pyi.in` package-data but no such templates exist;
   `pygpgpu/cuda/libraries/__init__.py:24` points at a README section that does not exist;
   `examples/` numbers two scripts `11_`; `docs/vec_type_after_table.txt` is a committed
   debug dump; the untracked `Mandelbrot.jpg` at the repo root duplicates
   `docs/Mandelbrot.jpg` (the one README references). Also: `Tape` resolves via
   `pygpgpu.Tape` but is in neither `__all__` nor `__dir__` — add it (the package's own
   convention is that an editor completes `pygpgpu.`).
6. **README gaps:** no gnp section, no host-libraries section, no mention of the numpy.array
   patch or the star-import shadowing; the `LICENSE` native-form author name is not among
   AGENTS.md's declared CJK exemptions (amend the policy or use the ASCII form there).

---

## Roadmap

### Now — before committing the AD stage
- P0 items 1-3 (each is a line-level fix plus one test).
- P1 item 8 (tape epochs for every gnp argument).
- Reconcile the AD status text (`docs/autodiff-plan.md`, the AGENTS.md header) and export
  `Tape` top-level.
- Keep the untracked `Mandelbrot.jpg` out of the commit.

### This release
- P1 items 4-7, 9, 10: buffer-claim `finally` on all backends, Metal binding/poller brackets,
  gemm dtype refusal, NVRTC cache fallback, OpenCL binary-path messages.
- P2 items 11-14: the import-cost fix (PEP 562 + deferred gnp/asyncio), no-device error
  aggregation, dead `_i` in generated CUDA, stderr traceback removal.
- Tooling: `check.py` stops mutating; ty ignores re-scoped; gate bugs fixed; doc-drift list
  cleared.

### Next release
- CI Tier 0, then Tier 1 as hardware allows.
- `LaunchPlan` declared vocabulary; `dup_defs.py` class-body extension; receiver-preamble
  extraction; `_dependency_order` moved down; slot-dtype table unified.
- Driver `Func` machinery extracted once (keep the two real hook points: lib-path discovery,
  preheat) — makes "error names the failing call" a one-line fix instead of five.
- Packaging/compliance: pcpp LICENSE + NOTICE; pygf guard; exception surface consolidated.
- Python floor decision (recommend >=3.9), with a py37 lint job only if 3.7 stays.

### Long term
- CUDA/HIP oop convergence to one implementation with per-backend difference points.
- Mixin shared-state contract (FunctionTranslator composes 40 mixins / ~495 distinct
  `self.*` attribute names with no declared contract between them).
- Evaluate isolating pygf's process-wide numpy.array patch.
- Wire `examples/` into pytest behind `gpu_backend`.

## Deliberately left alone — do not "fix" while working here

These are documented tradeoffs, listed so a cleanup pass does not undo them:

- OpenCL `set_data` gates on `CL_MEM_WRITE_ONLY` (`opencl/oop/buffer.py:145`) — OpenCL's API
  forbids uploading to a WRITE_ONLY buffer; the CUDA/HIP unification must not carry that
  gate over, and the reason should be commented at the site.
- The derived `global` work size may exceed the array length (bounds guard makes it safe) —
  do not restore exact-equality.
- The `python` backend executes bodies without translating — no AST codegen there.
- `domain=` is excluded from `_source_signature` on purpose (`pykernel_identity.py:41-47`).
- Passing an array where a scalar is declared is accepted on every backend (shared
  conversion) — Metal matches rather than diverging.
- The OpenCL 3-component vector classes are CUDA's 12-byte compact layout, OpenCL's are
  16-byte padded — this duplication is the ABI, not an accident.
- `_probe_backend` probing order is platform-dependent by design (`DEFAULT_BACKEND_ORDER`).
