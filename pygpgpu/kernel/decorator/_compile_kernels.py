from __future__ import annotations
import ast
import os

import numpy as np

from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
from .. import source_paths


# The numpy dtype a shared-buffer element travels as on Metal. It maps from the same
# translator spellings the byte-size table reads, so one `local_buffers` record serves
# both the launch (an element count and a dtype) and the byte size.
_LOCAL_DTYPES:Dict[str, np.dtype] = {
    "bool": np.dtype(np.bool_), "char": np.dtype(np.int8),
    "uchar": np.dtype(np.uint8), "short": np.dtype(np.int16),
    "ushort": np.dtype(np.uint16), "int": np.dtype(np.int32),
    "uint": np.dtype(np.uint32), "long": np.dtype(np.int64),
    "ulong": np.dtype(np.uint64), "float": np.dtype(np.float32),
    "double": np.dtype(np.float64),
}


def _local_dtype_for(ctype:str)->np.dtype:
    """The numpy dtype one shared-buffer element of `ctype` travels as on Metal."""
    return _LOCAL_DTYPES.get(ctype, np.dtype(np.float32))


def _random_sources_of(stage:object)->Tuple[str, ...]:
    """The host generators one translation unit drew from."""
    return tuple(sorted(getattr(stage, "random_sources", None) or ()))


def _merged_random_sources(translator:object, device_cache:Dict[str, object])->Tuple[str, ...]:
    """Every generator named by this kernel, its passes and the helpers it reaches.

    The launch's seed comes from these generators' states (see kernel/random.py), so a
    helper drawing from `np.random` must contribute that source even when the kernel body
    itself only called `random.random()`.
    """
    sources:Set[str] = set(_random_sources_of(translator))
    for stage in getattr(translator, "phase_translators", ()):
        sources.update(_random_sources_of(stage))
    for entry in device_cache.values():
        sources.update(entry.get("random_sources") or ())
    return tuple(sorted(sources))


def _merged_raise_sites(translator:object)->list:
    """Every raise site of this kernel, in the order the device ids will index them.

    A single-loop body has one translator; a body split into phases has one per phase, and
    the phases run in order -- each phase's kernel numbers its own sites from zero, so the
    ids have to be rebased onto the merged list for the host's lookup to land on the site
    that actually raised.
    """
    merged:list = []
    phase_translators = getattr(translator, "phase_translators", None) or []
    translators = phase_translators if phase_translators else [translator]
    for each in translators:
        sites = getattr(each, "raise_sites", None) or []
        if not sites:
            continue
        base = len(merged)
        for site in sites:
            entry = dict(site)
            entry["local_id"] = int(entry.get("id", 0))
            entry["id"] = base + entry["local_id"]
            merged.append(entry)
    return merged


class _CompileKernelsMixin:

    def _compile_kernels(self, plan:object, args:List[object], arg_types:List[object],
                    self_types:List[object], self_attrs:List[str],
                    owner:Optional[type], sig:str, backend:str, device_obj:object,
                    instance:object=None,
                    self_struct_members:Optional[Dict[str, str]]=None,
                    self_struct_type:object=None)->object:
        """The compilation itself, and the description of the launch it produces.

        The description is what a later call reads: the shapes, the work size, the
        return slots, and the per-phase metadata a multi-pass body needs.
        """
        # The whole file is one translation unit: a nested launch anywhere in it makes NVRTC
        # reject the source, so the *file* decides the build path, not the kernel being
        # compiled. A module with one nesting kernel and one ordinary kernel must be built
        # by nvcc even when the ordinary kernel is launched first, so the decision reads the
        # file that was actually written (it accumulates every kernel of the module).
        plan.linked:bool = plan.translator.nests_a_launch
        if not plan.linked:
            try:
                with open(plan.source_path, "r", encoding="utf-8") as plan.file:
                    plan.linked = "<<<" in plan.file.read()
            except OSError:
                plan.linked = False
        # Native recursion in the unit (a call site admitted to a self-call or a mutual
        # cycle) makes the per-thread stack a correctness parameter: CUDA/HIP give each
        # thread a small fixed stack by default, and a frame that spills past it writes
        # through memory nobody owns -- a wrong answer rather than an error. The launch
        # raises the stack limit before the first launch when the current one is too
        # small (see _raise_stack_for_recursion); the flag travels in the launch meta so
        # a cached entry keeps answering it.
        plan.recursive:bool = bool(getattr(plan.translator, "uses_native_recursion", False)
                             or any(getattr(phase, "uses_native_recursion", False)
                                    for phase in plan.translator.phase_translators))
        # The write above and the compile below are one critical section. A sibling kernel
        # of this same module may be appending its definition concurrently, and the compile
        # keys its cached binary on the file's *content*: a compile that started between two
        # appends would cache a binary of a half-populated file, and the kernel appended last
        # would find no entry point of its own in it (`KeyError` naming the kernel). Holding
        # the lock across both steps makes the file's content and the binary that describes
        # it change together.
        with source_paths.write_lock(plan.source_path):
            try:
                plan.kernels:List[object] = [
                    self._backend_compile(plan.source_path, plan.name, backend, device_obj, linked=plan.linked)
                    for plan.name, plan._ in plan.rendered]
            except KeyError:
                # A missing entry point: the unit itself was accepted (the build ran) but
                # does not export this kernel -- the stale-binary case, which the binary
                # cache's own mtime check already answers by rebuilding from source.
                # Rewriting the generated files here would discard the sibling kernels
                # other compiles of this module accumulated, so a KeyError reports as it is.
                raise
            except Exception:
                # A file can be damaged in ways the digest sidecar cannot see: text from a
                # previous package version whose generated shape changed (the forward-
                # declaration block did not exist before the owner-identity fix), a crashed
                # or truncated write, an indexer that saved a partial copy. The sidecar
                # describes the *python*, so all of these keep matching and the merge keeps
                # the damage forever. The one answer that is always right is the render this
                # process just built: rebuild every file this compilation writes from it
                # (headers, definitions and digests, discarding what the files held), then
                # retry once inside the same critical section. A build that fails again is
                # a real problem in the source or the backend, and its own error reports.
                plan.rewrite_plan = getattr(plan, "rewrite_plan", {}) or {}
                if plan.rewrite_plan:
                    for rewrite_path, (rewrite_header, rewrite_definitions, rewrite_python,
                                       rewrite_digest) in plan.rewrite_plan.items():
                        source_paths.rewrite_source(rewrite_path, rewrite_header,
                                                    rewrite_definitions, rewrite_python,
                                                    source_digest=rewrite_digest)
                    plan.kernels = [
                        self._backend_compile(plan.source_path, plan.name, backend,
                                              device_obj, linked=plan.linked)
                        for plan.name, plan._ in plan.rendered]
                else:
                    raise
        # A phase kernel declares only the shapes it indexes itself, so its launch needs its
        # own trailing extent arguments rather than the ones the first kernel wants.
        plan.extra_shapes:List[List[Tuple[int, int]]] = []
        # The created objects each phase declares: mid = np.zeros(4) between two loops is
        # a parameter of the pass that reads it and of no other.
        plan.phase_created:List[List[str]] = []
        # The per-phase parameter metadata (empty for a single-phase body, which never
        # enters _translate_phase_kernels; the fallback below fills it from the translator).
        # A gid-style body has no phases and no magic loop: its launch arguments are the
        # shared prefix plus the extents, which the fast path already builds, and inventing
        # a per-phase list here would double-count the outer-scope parameters.
        plan.is_magic_body:bool = bool(getattr(plan.translator, "phases", None)) and \
            len(getattr(plan.translator, "phases", [])) == 1 and \
            any(isinstance(stmt, ast.For) for stmt in plan.translator.tree.body)
        plan.phase_carried:List[List[str]] = []
        plan.phase_param_names:List[List[str]] = []
        # Per phase, whether its kernel declares the return claim/site words (see
        # render_return_params): a pass whose only return runs in its host tail does not.
        plan.phase_claims:List[bool] = []
        # Metal's per-phase launch metadata. A body split into passes gives each pass its
        # own kernel, and the shared buffers / barrier counter / atomic lock a signature
        # declares are facts about *that* signature -- `render.parameter_declarations`
        # reads them off the phase's own translator. Reading them off the main translator
        # instead left every such argument out of the launch for a multi-pass body (the
        # main translator visits no loop, so its flags are all False), which bound the
        # wrong values in every later slot.
        plan.phase_has_barrier:List[bool] = []
        plan.phase_has_atomic64:List[bool] = []
        plan.phase_local_counts:List[List[int]] = []
        plan.phase_local_dtypes:List[List[np.dtype]] = []
        if plan.translator.phase_translators:
            for plan.phase in plan.translator.phase_translators:
                plan.phase_index:Dict[str, int] = {
                    plan.name: position for position, (plan.name, plan._) in enumerate(plan.phase.params)}
                plan.extra_shapes.append([(plan.phase_index[array_name], dim)
                                     for array_name, dim in plan.phase.shapes_used])
                plan.phase_created.append(list(plan.phase.host_values))
                plan.phase_claims.append(bool(plan.phase._claims_on_device))
                plan.phase_has_barrier.append(bool(getattr(plan.phase, "has_barrier", False)))
                plan.phase_has_atomic64.append(bool(getattr(plan.phase, "has_atomic64", False)))
                plan.phase_local_counts.append([count for plan._, plan._t, count
                                                in plan.phase.local_buffers])
                plan.phase_local_dtypes.append([
                    plan.local_dtype_for(plan._t) for plan._, plan._t, plan._c
                    in plan.phase.local_buffers])
        else:
            # One magic loop: the main translator *is* the only phase, and it binds the
            # created objects on itself.
            plan.phase_index:Dict[str, int] = {
                plan.name: position for position, (plan.name, plan._) in enumerate(plan.translator.params)}
            plan.extra_shapes.append([(plan.phase_index[array_name], dim)
                                 for array_name, dim in plan.translator.shapes_used])
            plan.phase_created.append(list(plan.translator.host_values))
            # The one phase is the main translator: its parameters and carried slots are
            # the launch's, and the per-phase metadata would otherwise stay empty (the
            # launch then aligns against nothing and drops every argument after the first).
            if plan.is_magic_body:
                plan.phase_carried.append(list(plan.translator.carried_scalars))
                plan.phase_param_names.append(list(dict(plan.translator.params)))
                plan.phase_claims.append(bool(plan.translator._claims_on_device))

        # The launch description records what the generated files looked like when this
        # entry was built: one stat per file, keyed by the paths this compilation wrote.
        # A later launch consults it (see _source_files_changed) -- the digest sidecar
        # describes the *python*, so it cannot see a file that lost text on disk another
        # way, and a cached entry served from it would outlive the damage forever.
        plan.source_mtimes:Dict[str, int] = {}
        for plan.mtime_path in plan.rewrite_plan:
            try:
                plan.source_mtimes[plan.mtime_path] = os.stat(plan.mtime_path).st_mtime_ns
            except OSError:
                plan.source_mtimes[plan.mtime_path] = 0

        plan.launch:object = plan.kernels[0]

        # Atomic targets: static analysis cannot recognize pointer-cast writes, so force them to writable.
        # The set comes from the phase translators as well, and the flag is forced on every
        # kernel. A multi-pass body has **one kernel object per pass**, each with its own
        # argument table, and its loops live on the phase translators -- so reading only the
        # main translator's set (empty for such a body, because it never visits a loop) and
        # forcing only `kernels[0]` left a later pass's member store bound without the write
        # flag: the device wrote it and the host never read it back, so a member the kernel
        # assigned came home with its old value.
        from ...constants import KernelFlags
        plan.written_targets:Set[str] = set(plan.translator.atomic_targets)
        for plan.phase_translator in plan.translator.phase_translators:
            plan.written_targets.update(plan.phase_translator.atomic_targets)
        for plan.kernel in plan.kernels:
            for plan.target_name in plan.written_targets:
                if plan.target_name in plan.kernel._args:
                    plan.kernel._args[plan.target_name].kernel_flags = (
                        plan.kernel._args[plan.target_name].kernel_flags | KernelFlags.Writed)
            # The return slots are written by the device and read back on the host, so they
            # stay fully bidirectional. `atomicCAS(&_ret_claim[0], 0, 1)` *reads* the claim
            # word to compare it against zero, and a slot that was never uploaded would
            # compare against whatever the device buffer happened to hold -- the claim would
            # fire at random or never, and a launch would return the wrong site's value. That
            # is the same reason a write-only argument is still uploaded, applied to a slot
            # the library owns rather than to the caller's data.
            for plan.slot_name in (plan.translator.RETURN_CLAIM, plan.translator.RETURN_SITE):
                plan.slot_info:object = plan.kernel._args.get(plan.slot_name)
                if plan.slot_info is not None:
                    plan.slot_info.kernel_flags = (KernelFlags.Readed | KernelFlags.Writed)

        plan.param_index:Dict[str, int] = {plan.name: i for i, (plan.name, plan._) in enumerate(plan.translator.params)}
        # A multi-phase body is translated per phase, so the main translator indexes nothing;
        # the first phase's shapes are the ones this launch derives its work size from.
        plan.shapes_owner:object = (plan.translator.phase_translators[0]
                            if plan.translator.phase_translators else plan.translator)
        plan.shape_index:Dict[str, int] = {
            plan.name: position for position, (plan.name, plan._) in enumerate(plan.shapes_owner.params)}
        plan.shape_sources:List[Tuple[int, int]] = [
            (plan.shape_index[array_name], dim) for array_name, dim in plan.shapes_owner.shapes_used]
        if not plan.extra_shapes:
            # A single kernel: its own extents are the launch's.
            plan.extra_shapes = [list(plan.shape_sources)]
        # Each entry is (parameter index, dimension) for an array-sourced extent, or
        # (None, count) for a literal one (np.ndindex(2, 3)). A receiver-struct extent
        # is spelled `(self)->member` (or `(cls)->...` for a classmethod kernel) in the
        # translator's own spelling of the receiver name, so the parameter index is
        # looked up under that same name rather than a hard-coded "self".
        plan.self_param:str = (self._self_param if self._self_param else "self")
        plan.work_size_dims:List[Tuple[Optional[int], int]] = [
            (plan.param_index.get(plan.self_param), dim)
            if plan.name.startswith(f"({plan.self_param})->") else
            (plan.shape_index[plan.name], dim) if plan.name else (None, dim)
            for plan.name, dim in plan.shapes_owner.work_size_dims]
        # An `@kernel(domain=...)` declaration: expand each named array into its own
        # dimensions. The rank is only known at call time, so record (param, -1) markers and
        # let _invoke substitute the real per-dimension indices.
        if not plan.work_size_dims and plan.shapes_owner.declared_work_arrays:
            plan.work_size_dims = [(plan.shape_index[plan.name], -1)
                              for plan.name in plan.shapes_owner.declared_work_arrays]
        # local memory: opencl passes byte sizes at launch by parameter position; cuda uses __shared__ declarations and needs nothing passed
        plan.ctype_sizes:Dict[str, int] = {"bool": 1, "char": 1, "uchar": 1, "short": 2, "ushort": 2,
                                      "int": 4, "uint": 4, "long": 8, "ulong": 8,
                                      "float": 4, "double": 8}
        plan.local_sizes:List[int] = [plan.ctype_sizes.get(ctype, 4) * count for plan._, ctype, count in plan.translator.local_buffers]
        # Metal's shared buffers travel as device allocations (see `metal_codegen.barrier`
        # for why threadgroup memory is not used there): the launch needs the element
        # count per group and the element dtype, not a byte count. The numpy dtype maps
        # from the same table the byte sizes read, so one `local_buffers` record serves
        # both spellings.
        plan.local_dtype_for = _local_dtype_for
        plan.local_element_counts:List[int] = [count for plan._, ctype, count in plan.translator.local_buffers]
        plan.local_dtypes:List[np.dtype] = [
            _local_dtype_for(ctype) for plan._, ctype, count in plan.translator.local_buffers]
        # Images an OpenCL kernel samples need a sampler argument each; record which
        # parameter supplies the sampler so the launch can append it.
        plan.image_samplers:List[int] = [plan.param_index[plan.name]
                                    for plan.name in sorted(plan.translator.required_image_samplers)]
        # Return slots: one per distinct value type any return site writes, plus the
        # claim/site words. `arg_indices` maps an array a return site names to its
        # position in the launch arguments, so the result can hand back that same object.
        plan.returns:Dict[str, object] = {
            "slot_types": [plan.translator.return_slot_dtype(index)
                           for index in range(len(plan.translator.return_layout))],
            "slot_kinds": [entry["type"] for entry in plan.translator.return_layout],
            "sites": plan.translator.return_site_layout(),
            "arg_indices": {plan.name: position for plan.name, position in plan.param_index.items()},
        }
        return (plan.launch, plan.shape_sources, plan.work_size_dims, plan.local_sizes,
                plan.shapes_owner.uses_global_id, plan.image_samplers,
                plan.returns if plan.translator.has_returns else None,
                plan.translator.outer_vars,
                {"param_names": {position: plan.name for position, (plan.name, plan._) in enumerate(plan.translator.params)},
                 # Every kernel this body produced, in launch order: one per magic loop.
                 "carried": list(plan.translator.carried_scalars),
                 # Each carried slot's element type: the host allocates and reads the
                 # slot at this width (`total = 0.0` is a float slot, and an int32 host
                 # buffer under a `float*` parameter truncated every answer).
                 "carried_types": {plan.name: plan.t for plan.name, plan.t in
                                   ((plan.parameter_name, parameter_t[1])
                                    for plan.parameter_name, parameter_t in plan.translator.params
                                    if plan.parameter_name in plan.translator.carried_scalars
                                    and isinstance(parameter_t, tuple)
                                    and len(parameter_t) > 1)},
                 "gap_names": list(plan.translator.gap_names),
                 # The extra parameters each phase declares beyond the shared prefix: a gap
                 # value belongs to the pass that reads it (see _translate_phase_kernels).
                 "phase_carried": (plan.phase_carried if plan.phase_carried else
                                   [list(plan.phase.carried_scalars)
                                    for plan.phase in plan.translator.phase_translators]),
                 # The statements between two passes: the host runs region i between
                 # launch i and launch i+1 (see _run_host_region).
                 "phase_gaps": getattr(plan.translator, "phase_gaps", []),
                 # Each phase's parameters in order, so the launch aligns values to the
                 # signature (a phase may declare a carried name the first kernel does not).
                 "phase_param_names": (plan.phase_param_names if plan.phase_param_names else
                                       [list(dict(plan.phase.params))
                                        for plan.phase in plan.translator.phase_translators]),
                 # Whether each phase's kernel declares the return claim/site words. A
                 # single-phase body has one kernel (the main translator's) and no entry
                 # here: the launch then treats it as claiming, which its signature says.
                 "phase_claims": (plan.phase_claims if plan.phase_claims else
                                  [bool(plan.phase._claims_on_device)
                                   for plan.phase in plan.translator.phase_translators]),
                 # Whether the main kernel declares the claim/site words. The fast path
                 # launches that kernel with the argument list built once up front, so this
                 # has to describe the *first* kernel: a phase of a multi-loop body is one
                 # kernel of its own, while a single-loop body's only kernel is the main
                 # translator's (see render_return_params).
                 "main_claims": (bool(plan.translator.phase_translators[0]._claims_on_device)
                                 if plan.translator.phase_translators
                                 else bool(plan.translator._claims_on_device)),
                 # Whether each kernel declares Metal's spin-barrier counter, and the
                 # 64-bit atomic lock, and which shared buffers it expects. These are
                 # per-kernel facts: `parameter_declarations` reads them off the
                 # translator that renders *that* signature, so a multi-pass body's
                 # second pass may declare none while the first declares both. The
                 # single-phase fast path reads element 0 of each list.
                 "has_barrier": ([bool(getattr(stage, "has_barrier", False))
                                  for stage in plan.translator.phase_translators]
                                 if plan.translator.phase_translators
                                 else bool(getattr(plan.translator, "has_barrier", False))),
                 "has_atomic64": ([bool(getattr(stage, "has_atomic64", False))
                                   for stage in plan.translator.phase_translators]
                                  if plan.translator.phase_translators
                                  else bool(getattr(plan.translator, "has_atomic64", False))),
                 # Whether each kernel draws random numbers: such a kernel declares the
                 # launch seed as its last parameter, and the launch appends the seed value
                 # in that position (see render.random_seed_parameter and _build_phase_args).
                 "uses_random": ([bool(getattr(stage, "uses_random", False))
                                  for stage in plan.translator.phase_translators]
                                 if plan.translator.phase_translators
                                 else bool(getattr(plan.translator, "uses_random", False))),
                 # The host generators whose states seed this launch's stream: the
                 # kernel body, its passes and any device helper that draws.
                 "random_sources": _merged_random_sources(
                     plan.translator, plan.device_cache),
                 "local_element_counts": (plan.phase_local_counts
                                          if plan.translator.phase_translators
                                          else list(plan.local_element_counts)),
                 "local_dtypes": (plan.phase_local_dtypes
                                  if plan.translator.phase_translators
                                  else list(plan.local_dtypes)),
                 # Every raise the generated code contains, in source order. Its index (plus
                 # one) is the id the device writes into the exception slot, and the host
                 # rebuilds the class, message and line from this table after the launch
                 # (see _raise_from_exception_slot). Phases contribute their own sites, in
                 # the order the phases run, so the id the device writes always indexes the
                 # site that raised.
                 "raise_sites": _merged_raise_sites(plan.translator),
                 # The statements after the last magic loop, for a body split into passes:
                 # `body_tail` above belongs to a single-loop body, whose translation never
                 # enters _translate_phase_kernels. The last phase's own tail is the one the
                 # host runs after the final launch.
                 "phase_tail": (list(plan.translator.phase_translators[-1].body_tail)
                                if plan.translator.phase_translators else []),
                 # The receiver-struct members whose values are loop extents, in the
                 # order the -3 markers appear in work_size_dims (see _invoke's -3 branch).
                 # A multi-loop body's loops belong to the phase translators, so the
                 # members come from the same phase the extents came from.
                 "work_size_members": list(getattr(plan.shapes_owner, "work_size_members", [])),
                 # The statements after the last magic loop of a single-phase body: the host
                 # runs them once, after the launch (see _run_host_region).
                 "body_tail": getattr(plan.translator, "body_tail", []),
                 "kernels": plan.kernels,
                 # Whether the generated unit recurses natively: the launch raises the
                 # per-thread stack limit before launching (see _raise_stack_for_recursion).
                 "recursive": plan.recursive,
                 # The trailing extent arguments each phase kernel needs, by phase.
                 "kernel_shapes": plan.extra_shapes,
                 "kernel_created": plan.phase_created,
                 # Every created name any phase declares, so the host knows which objects
                 # the launch has to supply (a returned creation below the loop included).
                 "kernel_created_all": sorted({plan.name
                                               for names in plan.phase_created for plan.name in names}),
                 # The plain values a host-region statement bound, in the device-parameter
                 # order they were appended (a `w = len(label)` above the loop, an
                 # arbitrary-precision number the kernel receives as a scalar), and the
                 # values themselves for the launch to hand over.
                 "host_values": list(plan.translator.host_scalars),
                 "host_region_values": {plan.name: plan.value for plan.name, plan.value
                                        in self.host_region_snapshot.items()
                                        if plan.name in plan.translator.host_scalars},
                 "subscript_indices": dict(plan.translator.subscript_indices),
                 # name -> the widest *store* subscript, kept apart from the reads: a
                 # partial store is python's broadcast and is legal, so only the
                 # too-many direction is checked for it (see _check_shape_indices).
                 "subscript_store_indices": dict(plan.translator.subscript_store_indices),
                 # name -> the line of its widest subscript, and the offset that turns a
                 # parsed-copy line into the user's own: the launch blames that statement,
                 # not the loop, when a subscript names more indices than the rank.
                 "subscript_lines": dict(plan.translator.subscript_lines),
                 "subscript_columns": dict(plan.translator.subscript_columns),
                 # name -> the subscript as the user wrote it, so the frame stand-in can
                 # re-run that expression and let CPython draw its own caret.
                 "subscript_texts": dict(plan.translator.subscript_texts),
                 # The arrays a local was sliced from (`part = a[lo:hi]`): a slice is one
                 # pointer at an offset, which is only the same thing as python's view when
                 # the array has one dimension. Checked at launch, where the rank is known.
                 "sliced_arrays": list(plan.translator.sliced_arrays),
                 "source_first_line": int(getattr(plan.translator, "first_line", 1) or 1),
                 "loop_line": int(getattr(plan.translator, "magic_loop_line", 0) or 0),
                 # How many index names the magic loop unpacks: the launch checks it against
                 # the number of dimensions its iteration domain actually has.
                 "loop_names": len(plan.translator.work_item_indices),
                 # The array behind a `np.ndindex(arr.shape)` domain (index into the device
                 # parameter list, or -1), whose rank the loop must name one index per.
                 "whole_shape_index": plan.param_index.get(plan.translator.nd_whole_shape_param, -1),
                 # path -> st_mtime_ns of every generated file this entry was built from
                 # (see _source_files_changed, which answers "did a file change on disk").
                 "source_mtimes": dict(plan.source_mtimes)})
