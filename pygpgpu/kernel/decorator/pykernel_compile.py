# The compile: the translator run and the launch record, kept whole (its
# signature is pinned by nine test call sites); the cache helpers are
# pykernel_cache.py.
from __future__ import annotations
import ast
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
import numpy as np
from .. import structs
from ..translator import (
    DeviceType,
    FunctionTranslator,
)
from .. import source_paths
from .host_regions import (
    _called_modules,
    _dependency_order,
    _host_created_values,
    _positional_argument_names,
)
from .factory import (
    kernel,
)


class _PykernelCompileMixin:

    def _compile(self, args:List[object], arg_types:List[object], self_types:List[object],
                 self_attrs:List[str], owner:Optional[type], sig:str,
                 backend:str, device_obj:object, instance:object=None,
                 self_struct_members:Optional[Dict[str, str]]=None,
                 self_struct_type:object=None)->Tuple[object, List[str]]:
        if backend in ("cuda", "hip"):
            for t in list(arg_types) + list(self_types):
                base:str = (t[1] if isinstance(t, tuple) else t)
                if base == "half":
                    raise SyntaxError(
                        "fp16 (np.float16) arguments are not supported on the cuda backend: "
                        "NVRTC on this machine cannot resolve cuda_fp16.h; use np.float32 instead") from None

        c_name:str = self._c_kernel_name(sig, backend)
        device_cache:Dict[object, object] = {}
        # self_types leads with the receiver struct when the receiver travels as one, so
        # the attribute map pairs the array attributes with the types after it.
        self_attr_types:Dict[str, object] = dict(zip(
            self_attrs, self_types[1:] if self_struct_members else self_types))

        # hip accepts CUDA syntax (threadIdx/atomicAdd/make_float4 etc.), so codegen goes through cuda
        codegen_backend:str = ("cuda" if backend == "hip" else backend)
        # Statements above the magic loop that only build data are executed by the
        # interpreter, not translated: numpy decides the shape and dtype, and the resulting
        # object is what the kernel receives as a parameter (see
        # FunctionTranslator._host_created_assignment). The values and the line numbers were
        # computed once for this call, before the body was scanned (see _invoke_inner).
        creation_values, _ = _host_created_values(self.raw_func, args, instance)
        executed_lines:List[int] = list(getattr(self, "host_region_statements", []))
        region_values:Dict[str, object] = dict(
            getattr(self, "host_region_snapshot", None) or creation_values)
        # The class each STRUCTURE argument was actually built from. One dtype can stand
        # for several classes, so the dtype alone cannot say whose method `p.total` is --
        # but the launch has the instance the caller handed over, and its type is the
        # answer. Keyed by parameter name, which is what the translator's expressions carry.
        # `structure_dtype_of` excludes the vector types, which are ctypes classes too.
        struct_arg_classes:Dict[str, type] = {}
        parameter_names:List[str] = _positional_argument_names(self.raw_func)
        for parameter_name, value in zip(parameter_names, args):
            # A union is a `ctypes` record as well but not a `Structure` subclass, so the
            # class is recognized through the shared predicate rather than `ctypes.Structure`.
            if not structs.is_ctypes_record(value):
                continue
            # structure_dtype_of excludes the vector types, which are ctypes classes too.
            if structs.structure_dtype_of(value) is None:
                continue
            struct_arg_classes[parameter_name] = type(value)
        translator:FunctionTranslator = FunctionTranslator(
            self.raw_func, list(arg_types) + list(self_types), codegen_backend,
            self._resolve_global, device_cache, is_kernel=True,
            self_param=self._self_param, self_attr_types=self_attr_types,
            self_owner=owner, launch_backend=backend,
            self_struct_members=self_struct_members,
            self_struct_type=self_struct_type,
            struct_arg_classes=struct_arg_classes,
            host_creation_values=region_values,
            host_executed_lines=executed_lines,
            type_map=self._type_map)
        # `@kernel(domain="x")` names the parameter whose shape declares the iteration
        # domain. It is applied before `translate()` so the declaration is in place while the
        # magic loop is scanned -- a body cannot carry one of its own any more.
        if self._domain:
            self._apply_declared_domain(translator)
        translator.translate()
        # The name has to cover what the translator actually declares. `sig` is built from
        # the *argument types* the caller passed, which do not include the extent parameters
        # derived from the body, the outer-scope values or the shared buffers -- so a body
        # whose derived signature changed kept its old name, and the generated file ended up
        # holding two declarations of it ("more than one instance of overloaded function").
        # Re-deriving the name with the rendered declaration folds that into the same hash.
        c_name = self._c_kernel_name(sig, backend,
                                     declaration=";".join(translator.parameter_declarations()))
        # One kernel per magic loop: a body with several renders several, launched in order
        # so the arrays stay on the device between them.
        rendered:List[Tuple[str, str]] = translator.render_kernels(c_name)
        kernel_code:str = "\n\n".join(source for _, source in rendered)

        # The generated device code is split the way the python code is (see source_paths):
        # a kernel and every helper written in its own module go to that module's source
        # file, and a helper from another module goes to that module's file and is reached
        # with one `#include`. One file per python module means one compilation unit per
        # module, so a callee from another module still has to be reachable -- which is what
        # the include does (the preprocessor expands it before the compiler runs).
        kernel_source_file:str = getattr(self.raw_func.__code__, "co_filename", "") or ""
        # What the generated file records so a later call can tell whether it still
        # describes this module's text. mtime alone is not reliable (`git checkout` restores
        # the old stamp), so the digest is what actually decides.
        source_digest:str = source_paths.python_source_digest(kernel_source_file)

        def module_of(entry:Dict[str, object])->str:
            """The python file a device function came from, spelled as a cache path."""
            return source_paths.source_path(kernel_source_file, entry.get("source_file", ""),
                                            backend)

        # Device helper functions are emitted in dependency order: C requires a callee to be
        # declared before its caller, and the cache is filled depth-first (a nested helper
        # is finished *after* the helper that calls it). Emit leaves first.
        ordered_entries:List[Dict[str, object]] = _dependency_order(device_cache)
        own_entries:List[Dict[str, object]] = [
            entry for entry in ordered_entries
            if module_of(entry) == module_of({"source_file": kernel_source_file})]
        foreign_entries:Dict[str, List[Dict[str, object]]] = {}
        for entry in ordered_entries:
            path:str = module_of(entry)
            if path == module_of({"source_file": kernel_source_file}):
                continue
            foreign_entries.setdefault(path, []).append(entry)

        # A nested launch needs the callee's own `__global__` function in this same source:
        # the launch statement names a device function, and C requires it before its caller.
        # A built-in that a backend spells with a helper contributes one too, and it is a
        # device function the kernel calls, so it goes in the same place.
        device_prefix:List[str] = (list(translator.device_helpers)
                                   + list(translator.nested_helpers)
                                   + list(translator.nested_kernels))
        device_prefix.extend(entry["code"] for entry in own_entries)

        structure_dtypes:List[DeviceType] = list(translator.structure_dtypes())
        for entry in device_cache.values():
            for dtype in entry.get("structs", ()):
                if dtype not in structure_dtypes:
                    structure_dtypes.append(dtype)
        # Every structure this compilation reaches is declared by the kernel's own file. A
        # helper in another module names its types too, and that module's file is reached
        # from here, so the declaration has to be in scope before the include is read --
        # putting it in the helper's own file instead made a cycle of includes and, when two
        # modules shared one structure, the same declaration twice in one file.
        typedefs:str = (structs.render_structures(structure_dtypes, translator.backend)
                        if structure_dtypes else "")
        # A structure member whose type a fixed header declares needs that header even when
        # the body never operates on the value: `struct S { c64 z; };` names `c64`, so the
        # declaration has to be in scope. Walking the member dtypes here is what covers that
        # case -- the translator only records a width when an operation uses one.
        from .. import complex as complex_support  # noqa: PLC0415 (import cycle)
        from .. import quaternion as quaternion_support  # noqa: PLC0415 (import cycle)

        member_quaternion_needs:Set[str] = set()
        member_complex_needs:Set[str] = set()
        for dtype in structure_dtypes:
            record:np.dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
            for name in record.names or ():
                field_dtype:np.dtype = np.dtype(record.fields[name][0])
                if field_dtype.subdtype is not None:
                    field_dtype = field_dtype.subdtype[0]
                quat_need:Optional[str] = quaternion_support.needs(field_dtype)
                if quat_need is not None:
                    member_quaternion_needs.add(quat_need)
                complex_need:Optional[str] = complex_support.needs(field_dtype)
                if complex_need is not None:
                    member_complex_needs.add(complex_need)

        # Quaternion structures and their arithmetic both come from a fixed header, copied
        # into the cache's include folder beside the generated source. The source needs only
        # the enable macros for the widths in use, then one include -- no declaration of its
        # own, because the header spells the two type names in advance.
        quaternion_needs:Set[str] = set(translator.quaternion_needs)
        # A member declared by the fixed header needs it even when the body never
        # operates on the value (see the walk above).
        quaternion_needs.update(member_quaternion_needs)
        for entry in device_cache.values():
            quaternion_needs.update(entry.get("quaternion_needs") or ())
        include_lines:str = ""
        if quaternion_needs:
            from .. import quaternion as quaternion_header

            include_lines = quaternion_header.include_line(
                tuple(sorted(quaternion_needs)), backend, self._cache_dir(backend))

        # Complex values work the same way, from their own fixed header: the two structures
        # and every operation on them are declared there, so a source that includes it says
        # nothing else about a complex number. A helper's blocks travel with its cache entry,
        # exactly as a helper's quaternion widths do.
        complex_needs:Set[str] = set(translator.complex_needs)
        complex_needs.update(member_complex_needs)
        for entry in device_cache.values():
            complex_needs.update(entry.get("complex_needs") or ())
        if complex_needs:
            from .. import complex as complex_header

            complex_lines:str = complex_header.include_line(
                tuple(sorted(complex_needs)), backend, self._cache_dir(backend))
            include_lines = (f"{include_lines}\n{complex_lines}" if include_lines
                             else complex_lines)

        # A registered device-side library function may need headers the compilation does
        # not have yet. A header the user ships as a *path* is copied into the cache's
        # include folder and included by name; a header given as a full `#include` line
        # (a system header, say) is carried into the generated source verbatim.
        library_header_lines:List[str] = []
        if translator.device_library_headers:
            from ..device_library import include_lines_for

            library_header_lines = include_lines_for(
                backend, translator.device_library_headers, self._cache_dir(backend))
        if library_header_lines:
            library_includes:str = "\n".join(library_header_lines)
            include_lines = (f"{include_lines}\n{library_includes}" if include_lines
                             else library_includes)

        # Every top-level definition is separated by one blank line: helper from helper,
        # helper from kernel. A trailing blank on each piece makes the join produce exactly
        # one, and the final trim drops the one after the last definition.
        pieces:List[str] = [piece.rstrip("\n") for piece in device_prefix]
        pieces.append(kernel_code.rstrip("\n"))

        own_path:str = module_of({"source_file": kernel_source_file})
        # The includes come first: a callee from another module is a declaration this file
        # needs before anything that calls it. The order is the order the callees were
        # emitted, so a foreign helper that itself includes another is already covered.
        # A structure declared by another module: that module's file holds the `struct`,
        # and this file includes it rather than repeating the declaration. A structure with
        # no python class behind it stays with the file that uses it.
        # The includes are ordered the same way the definitions are: a module whose code
        # calls another module's function has to be reached after that callee, because C
        # requires a function before its caller. The definitions are already in dependency
        # order, so the modules are taken in the order their first definition appears.
        ordered_foreign:List[str] = []
        for entry in ordered_entries:
            path:str = module_of(entry)
            if path in foreign_entries and path not in ordered_foreign:
                ordered_foreign.append(path)
        foreign_includes:List[str] = [
            source_paths.include_line(own_path, path) for path in ordered_foreign]
        # A generated file is a *header* plus a list of definitions. The header names every
        # structure the file's own code uses and includes every other module it reaches;
        # the definitions are appended one at a time, so a module with several kernels keeps
        # all of them in one file. The header comes first because C needs a type before the
        # function that names it, and a callee before its caller.
        own_header:List[str] = []
        if backend == "metal":
            # Metal's prelude: the `#include <metal_stdlib>` and the macros that give the
            # shared body its OpenCL index spellings (see kernel/metal_codegen.py). It
            # precedes the typedefs because the macros are the first thing the kernel body
            # needs -- and it must be first in the *file*, before any declaration.
            from ..metal_codegen import prelude
            own_header.append(prelude())
        if include_lines:
            # The fixed headers come BEFORE the typedef block: a structure member can name
            # `quatf` or `c64`, and that declaration has to be in scope when the structure
            # is declared, not after it.
            own_header.append(include_lines)
        if typedefs:
            own_header.append(typedefs)
        if foreign_includes:
            own_header.append("\n".join(foreign_includes))

        written:Dict[str, str] = {}
        for path, entries in foreign_entries.items():
            # A foreign module's file declares no structures of its own. The kernel's file
            # carries the whole typedef block ahead of the includes that reach these files,
            # and each declaration is guarded by its structure name, so the type is declared
            # once no matter how many files are compiled from it. Repeating the block here
            # would put every declaration in every file, which is a cycle of includes and a
            # duplicate definition in the one file two modules share.
            foreign_header:List[str] = []
            if include_lines:
                foreign_header.append(include_lines)
            # A foreign module may itself call into a third module: every structure is
            # declared by the top-level file, but a *function* it calls has to be reached
            # from its own file, and only by the files that call it.
            callee_includes:List[str] = [
                source_paths.include_line(path, other) for other in ordered_foreign
                if other != path and other in _called_modules(path, foreign_entries)]
            if callee_includes:
                foreign_header.append("\n".join(callee_includes))
            written[path] = source_paths.append_source(
                path, foreign_header,
                [entry["code"] for entry in entries if entry["code"]],
                entries[0].get("source_file", "") or "",
                source_digest=source_digest)

        source_path:str = source_paths.append_source(
            own_path, own_header, [piece for piece in pieces if piece],
            kernel_source_file, source_digest=source_digest)
        # The whole file is one translation unit: a nested launch anywhere in it makes NVRTC
        # reject the source, so the *file* decides the build path, not the kernel being
        # compiled. A module with one nesting kernel and one ordinary kernel must be built
        # by nvcc even when the ordinary kernel is launched first, so the decision reads the
        # file that was actually written (it accumulates every kernel of the module).
        linked:bool = translator.nests_a_launch
        if not linked:
            try:
                with open(source_path, "r", encoding="utf-8") as file:
                    linked = "<<<" in file.read()
            except OSError:
                linked = False
        # The write above and the compile below are one critical section. A sibling kernel
        # of this same module may be appending its definition concurrently, and the compile
        # keys its cached binary on the file's *content*: a compile that started between two
        # appends would cache a binary of a half-populated file, and the kernel appended last
        # would find no entry point of its own in it (`KeyError` naming the kernel). Holding
        # the lock across both steps makes the file's content and the binary that describes
        # it change together.
        with source_paths.write_lock(source_path):
            kernels:List[object] = [
                self._backend_compile(source_path, name, backend, device_obj, linked=linked)
                for name, _ in rendered]
        # A phase kernel declares only the shapes it indexes itself, so its launch needs its
        # own trailing extent arguments rather than the ones the first kernel wants.
        extra_shapes:List[List[Tuple[int, int]]] = []
        # The created objects each phase declares: mid = np.zeros(4) between two loops is
        # a parameter of the pass that reads it and of no other.
        phase_created:List[List[str]] = []
        # The per-phase parameter metadata (empty for a single-phase body, which never
        # enters _translate_phase_kernels; the fallback below fills it from the translator).
        # A gid-style body has no phases and no magic loop: its launch arguments are the
        # shared prefix plus the extents, which the fast path already builds, and inventing
        # a per-phase list here would double-count the outer-scope parameters.
        is_magic_body:bool = bool(getattr(translator, "phases", None)) and \
            len(getattr(translator, "phases", [])) == 1 and \
            any(isinstance(stmt, ast.For) for stmt in translator.tree.body)
        phase_carried:List[List[str]] = []
        phase_param_names:List[List[str]] = []
        # Per phase, whether its kernel declares the return claim/site words (see
        # render_return_params): a pass whose only return runs in its host tail does not.
        phase_claims:List[bool] = []
        if translator.phase_translators:
            for phase in translator.phase_translators:
                phase_index:Dict[str, int] = {
                    name: position for position, (name, _) in enumerate(phase.params)}
                extra_shapes.append([(phase_index[array_name], dim)
                                     for array_name, dim in phase.shapes_used])
                phase_created.append(list(phase.host_values))
                phase_claims.append(bool(phase._claims_on_device))
        else:
            # One magic loop: the main translator *is* the only phase, and it binds the
            # created objects on itself.
            phase_index:Dict[str, int] = {
                name: position for position, (name, _) in enumerate(translator.params)}
            extra_shapes.append([(phase_index[array_name], dim)
                                 for array_name, dim in translator.shapes_used])
            phase_created.append(list(translator.host_values))
            # The one phase is the main translator: its parameters and carried slots are
            # the launch's, and the per-phase metadata would otherwise stay empty (the
            # launch then aligns against nothing and drops every argument after the first).
            if is_magic_body:
                phase_carried.append(list(translator.carried_scalars))
                phase_param_names.append(list(dict(translator.params)))
                phase_claims.append(bool(translator._claims_on_device))

        launch:object = kernels[0]

        # Atomic targets: static analysis cannot recognize pointer-cast writes, so force them to writable
        from ...constants import KernelFlags
        for target_name in translator.atomic_targets:
            if target_name in launch._args:
                launch._args[target_name].kernel_flags = (launch._args[target_name].kernel_flags | KernelFlags.Writed)

        # The return slots are written by the device and read back on the host, so they stay
        # fully bidirectional. `atomicCAS(&_ret_claim[0], 0, 1)` *reads* the claim word to
        # compare it against zero, and a slot that was never uploaded would compare against
        # whatever the device buffer happened to hold -- the claim would fire at random or
        # never, and a launch would return the wrong site's value. That is the same reason a
        # write-only argument is still uploaded, applied to a slot the library owns rather
        # than to the caller's data.
        for slot_name in (translator.RETURN_CLAIM, translator.RETURN_SITE):
            slot_info:object = launch._args.get(slot_name)
            if slot_info is not None:
                slot_info.kernel_flags = (KernelFlags.Readed | KernelFlags.Writed)

        param_index:Dict[str, int] = {name: i for i, (name, _) in enumerate(translator.params)}
        # A multi-phase body is translated per phase, so the main translator indexes nothing;
        # the first phase's shapes are the ones this launch derives its work size from.
        shapes_owner:object = (translator.phase_translators[0]
                            if translator.phase_translators else translator)
        shape_index:Dict[str, int] = {
            name: position for position, (name, _) in enumerate(shapes_owner.params)}
        shape_sources:List[Tuple[int, int]] = [
            (shape_index[array_name], dim) for array_name, dim in shapes_owner.shapes_used]
        if not extra_shapes:
            # A single kernel: its own extents are the launch's.
            extra_shapes = [list(shape_sources)]
        # Each entry is (parameter index, dimension) for an array-sourced extent, or
        # (None, count) for a literal one (np.ndindex(2, 3)).
        work_size_dims:List[Tuple[Optional[int], int]] = [
            (param_index.get("self"), dim)
            if name.startswith("(self)->") else
            (shape_index[name], dim) if name else (None, dim)
            for name, dim in shapes_owner.work_size_dims]
        # An `@kernel(domain=...)` declaration: expand each named array into its own
        # dimensions. The rank is only known at call time, so record (param, -1) markers and
        # let _invoke substitute the real per-dimension indices.
        if not work_size_dims and shapes_owner.declared_work_arrays:
            work_size_dims = [(shape_index[name], -1)
                              for name in shapes_owner.declared_work_arrays]
        # local memory: opencl passes byte sizes at launch by parameter position; cuda uses __shared__ declarations and needs nothing passed
        ctype_sizes:Dict[str, int] = {"bool": 1, "char": 1, "uchar": 1, "short": 2, "ushort": 2,
                                      "int": 4, "uint": 4, "long": 8, "ulong": 8,
                                      "float": 4, "double": 8}
        local_sizes:List[int] = [ctype_sizes.get(ctype, 4) * count for _, ctype, count in translator.local_buffers]
        # Images an OpenCL kernel samples need a sampler argument each; record which
        # parameter supplies the sampler so the launch can append it.
        image_samplers:List[int] = [param_index[name]
                                    for name in sorted(translator.required_image_samplers)]
        # Return slots: one per distinct value type any return site writes, plus the
        # claim/site words. `arg_indices` maps an array a return site names to its
        # position in the launch arguments, so the result can hand back that same object.
        returns:Dict[str, object] = {
            "slot_types": [translator.return_slot_dtype(index)
                           for index in range(len(translator.return_layout))],
            "slot_kinds": [entry["type"] for entry in translator.return_layout],
            "sites": translator.return_site_layout(),
            "arg_indices": {name: position for name, position in param_index.items()},
        }
        return (kernel, shape_sources, work_size_dims, local_sizes,
                shapes_owner.uses_global_id, image_samplers,
                returns if translator.has_returns else None,
                translator.outer_vars,
                {"param_names": {position: name for name, position in param_index.items()},
                 # Every kernel this body produced, in launch order: one per magic loop.
                 "carried": list(translator.carried_scalars),
                 # Each carried slot's element type: the host allocates and reads the
                 # slot at this width (`total = 0.0` is a float slot, and an int32 host
                 # buffer under a `float*` parameter truncated every answer).
                 "carried_types": {name: t for name, t in
                                   ((parameter_name, parameter_t[1])
                                    for parameter_name, parameter_t in translator.params
                                    if parameter_name in translator.carried_scalars
                                    and isinstance(parameter_t, tuple)
                                    and len(parameter_t) > 1)},
                 "gap_names": list(translator.gap_names),
                 # The extra parameters each phase declares beyond the shared prefix: a gap
                 # value belongs to the pass that reads it (see _translate_phase_kernels).
                 "phase_carried": (phase_carried if phase_carried else
                                   [list(phase.carried_scalars)
                                    for phase in translator.phase_translators]),
                 # The statements between two passes: the host runs region i between
                 # launch i and launch i+1 (see _run_host_region).
                 "phase_gaps": getattr(translator, "phase_gaps", []),
                 # Each phase's parameters in order, so the launch aligns values to the
                 # signature (a phase may declare a carried name the first kernel does not).
                 "phase_param_names": (phase_param_names if phase_param_names else
                                       [list(dict(phase.params))
                                        for phase in translator.phase_translators]),
                 # Whether each phase's kernel declares the return claim/site words. A
                 # single-phase body has one kernel (the main translator's) and no entry
                 # here: the launch then treats it as claiming, which its signature says.
                 "phase_claims": (phase_claims if phase_claims else
                                  [bool(phase._claims_on_device)
                                   for phase in translator.phase_translators]),
                 # Whether the main kernel declares the claim/site words. The fast path
                 # launches that kernel with the argument list built once up front, so this
                 # has to describe the *first* kernel: a phase of a multi-loop body is one
                 # kernel of its own, while a single-loop body's only kernel is the main
                 # translator's (see render_return_params).
                 "main_claims": (bool(translator.phase_translators[0]._claims_on_device)
                                 if translator.phase_translators
                                 else bool(translator._claims_on_device)),
                 # The statements after the last magic loop, for a body split into passes:
                 # `body_tail` above belongs to a single-loop body, whose translation never
                 # enters _translate_phase_kernels. The last phase's own tail is the one the
                 # host runs after the final launch.
                 "phase_tail": (list(translator.phase_translators[-1].body_tail)
                                if translator.phase_translators else []),
                 # The receiver-struct members whose values are loop extents, in the
                 # order the -3 markers appear in work_size_dims (see _invoke's -3 branch).
                 "work_size_members": list(getattr(translator, "work_size_members", [])),
                 # The statements after the last magic loop of a single-phase body: the host
                 # runs them once, after the launch (see _run_host_region).
                 "body_tail": getattr(translator, "body_tail", []),
                 "kernels": kernels,
                 # The trailing extent arguments each phase kernel needs, by phase.
                 "kernel_shapes": extra_shapes,
                 "kernel_created": phase_created,
                 # Every created name any phase declares, so the host knows which objects
                 # the launch has to supply (a returned creation below the loop included).
                 "kernel_created_all": sorted({name
                                               for names in phase_created for name in names}),
                 # The plain values a host-region statement bound, in the device-parameter
                 # order they were appended (a `w = len(label)` above the loop, an
                 # arbitrary-precision number the kernel receives as a scalar), and the
                 # values themselves for the launch to hand over.
                 "host_values": list(translator.host_scalars),
                 "host_region_values": {name: value for name, value
                                        in self.host_region_snapshot.items()
                                        if name in translator.host_scalars},
                 "subscript_indices": dict(translator.subscript_indices),
                 # name -> the line of its widest subscript, and the offset that turns a
                 # parsed-copy line into the user's own: the launch blames that statement,
                 # not the loop, when a subscript names more indices than the rank.
                 "subscript_lines": dict(translator.subscript_lines),
                 "subscript_columns": dict(translator.subscript_columns),
                 # name -> the subscript as the user wrote it, so the frame stand-in can
                 # re-run that expression and let CPython draw its own caret.
                 "subscript_texts": dict(translator.subscript_texts),
                 # The arrays a local was sliced from (`part = a[lo:hi]`): a slice is one
                 # pointer at an offset, which is only the same thing as python's view when
                 # the array has one dimension. Checked at launch, where the rank is known.
                 "sliced_arrays": list(translator.sliced_arrays),
                 "source_first_line": int(getattr(translator, "first_line", 1) or 1),
                 "loop_line": int(getattr(translator, "magic_loop_line", 0) or 0),
                 # How many index names the magic loop unpacks: the launch checks it against
                 # the number of dimensions its iteration domain actually has.
                 "loop_names": len(translator.work_item_indices),
                 # The array behind a `np.ndindex(arr.shape)` domain (index into the device
                 # parameter list, or -1), whose rank the loop must name one index per.
                 "whole_shape_index": param_index.get(translator.nd_whole_shape_param, -1)})
