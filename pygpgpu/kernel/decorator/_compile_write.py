from __future__ import annotations
from typing import (
    Dict,
    List,
    Optional,
    Set,
)
import numpy as np
from .. import structs
from ..translator import (
    DeviceType,
)
from .. import source_paths
from .host_regions import (
    _called_modules,
    _dependency_order,
)


class _CompileWriteMixin:

    def _write_source(self, plan:object, args:List[object], arg_types:List[object],
                    self_types:List[object], self_attrs:List[str],
                    owner:Optional[type], sig:str, backend:str, device_obj:object,
                    instance:object=None,
                    self_struct_members:Optional[Dict[str, str]]=None,
                    self_struct_type:object=None)->object:
        """The generated file: its header, its definitions, and the includes that
        reach the other modules.

        The generated device code is split the way the python code is (see
        source_paths): a kernel and every helper written in its own module go to that
        module's source file, and a helper from another module goes to that module's
        file and is reached with one `#include`.
        """
        # The generated device code is split the way the python code is (see source_paths):
        # a kernel and every helper written in its own module go to that module's source
        # file, and a helper from another module goes to that module's file and is reached
        # with one `#include`. One file per python module means one compilation unit per
        # module, so a callee from another module still has to be reachable -- which is what
        # the include does (the preprocessor expands it before the compiler runs).
        plan.kernel_source_file:str = getattr(self.raw_func.__code__, "co_filename", "") or ""
        # What the generated file records so a later call can tell whether it still
        # describes this module's text. mtime alone is not reliable (`git checkout` restores
        # the old stamp), so the digest is what actually decides.
        plan.source_digest:str = source_paths.python_source_digest(plan.kernel_source_file)

        def module_of(entry:Dict[str, object])->str:
            """The python file a device function came from, spelled as a cache path."""
            return source_paths.source_path(plan.kernel_source_file, entry.get("source_file", ""),
                                            backend)

        # Device helper functions are emitted in dependency order: C requires a callee to be
        # declared before its caller, and the cache is filled depth-first (a nested helper
        # is finished *after* the helper that calls it). Emit leaves first.
        plan.ordered_entries:List[Dict[str, object]] = _dependency_order(plan.device_cache)
        plan.own_entries:List[Dict[str, object]] = [
            entry for entry in plan.ordered_entries
            if module_of(entry) == module_of({"source_file": plan.kernel_source_file})]
        plan.foreign_entries:Dict[str, List[Dict[str, object]]] = {}
        for entry in plan.ordered_entries:
            plan.path:str = module_of(entry)
            if plan.path == module_of({"source_file": plan.kernel_source_file}):
                continue
            plan.foreign_entries.setdefault(plan.path, []).append(entry)

        # A nested launch needs the callee's own `__global__` function in this same source:
        # the launch statement names a device function, and C requires it before its caller.
        # A built-in that a backend spells with a helper contributes one too, and it is a
        # device function the kernel calls, so it goes in the same place.
        plan.device_prefix:List[str] = (list(plan.translator.device_helpers)
                                   + list(plan.translator.nested_helpers)
                                   + list(plan.translator.nested_kernels))
        plan.device_prefix.extend(entry["code"] for entry in plan.own_entries)

        plan.structure_dtypes:List[DeviceType] = list(plan.translator.structure_dtypes())
        for entry in plan.device_cache.values():
            for plan.dtype in entry.get("structs", ()):
                if plan.dtype not in plan.structure_dtypes:
                    plan.structure_dtypes.append(plan.dtype)
        # Every structure this compilation reaches is declared by the kernel's own file. A
        # helper in another module names its types too, and that module's file is reached
        # from here, so the declaration has to be in scope before the include is read --
        # putting it in the helper's own file instead made a cycle of includes and, when two
        # modules shared one structure, the same declaration twice in one file.
        plan.typedefs:str = (structs.render_structures(plan.structure_dtypes, plan.translator.backend)
                        if plan.structure_dtypes else "")
        # A structure member whose type a fixed header declares needs that header even when
        # the body never operates on the value: `struct S { c64 z; };` names `c64`, so the
        # declaration has to be in scope. Walking the member dtypes here is what covers that
        # case -- the translator only records a width when an operation uses one.
        from .. import complex as complex_support  # noqa: PLC0415 (import cycle)
        from .. import quaternion as quaternion_support  # noqa: PLC0415 (import cycle)

        plan.member_quaternion_needs:Set[str] = set()
        plan.member_complex_needs:Set[str] = set()
        for plan.dtype in plan.structure_dtypes:
            plan.record:np.dtype = plan.dtype if isinstance(plan.dtype, np.dtype) else np.dtype(plan.dtype)
            for plan.name in plan.record.names or ():
                plan.field_dtype:np.dtype = np.dtype(plan.record.fields[plan.name][0])
                if plan.field_dtype.subdtype is not None:
                    plan.field_dtype = plan.field_dtype.subdtype[0]
                plan.quat_need:Optional[str] = quaternion_support.needs(plan.field_dtype)
                if plan.quat_need is not None:
                    plan.member_quaternion_needs.add(plan.quat_need)
                plan.complex_need:Optional[str] = complex_support.needs(plan.field_dtype)
                if plan.complex_need is not None:
                    plan.member_complex_needs.add(plan.complex_need)

        # Quaternion structures and their arithmetic both come from a fixed header, copied
        # into the cache's include folder beside the generated source. The source needs only
        # the enable macros for the widths in use, then one include -- no declaration of its
        # own, because the header spells the two type names in advance.
        plan.quaternion_needs:Set[str] = set(plan.translator.quaternion_needs)
        # A member declared by the fixed header needs it even when the body never
        # operates on the value (see the walk above).
        plan.quaternion_needs.update(plan.member_quaternion_needs)
        for entry in plan.device_cache.values():
            plan.quaternion_needs.update(entry.get("quaternion_needs") or ())
        plan.include_lines:str = ""
        if plan.quaternion_needs:
            from .. import quaternion as quaternion_header

            plan.include_lines = quaternion_header.include_line(
                tuple(sorted(plan.quaternion_needs)), backend, self._cache_dir(backend))

        # Complex values work the same way, from their own fixed header: the two structures
        # and every operation on them are declared there, so a source that includes it says
        # nothing else about a complex number. A helper's blocks travel with its cache entry,
        # exactly as a helper's quaternion widths do.
        plan.complex_needs:Set[str] = set(plan.translator.complex_needs)
        plan.complex_needs.update(plan.member_complex_needs)
        for entry in plan.device_cache.values():
            plan.complex_needs.update(entry.get("complex_needs") or ())
        if plan.complex_needs:
            from .. import complex as complex_header

            plan.complex_lines:str = complex_header.include_line(
                tuple(sorted(plan.complex_needs)), backend, self._cache_dir(backend))
            plan.include_lines = (f"{plan.include_lines}\n{plan.complex_lines}" if plan.include_lines
                             else plan.complex_lines)

        # A registered device-side library function may need headers the compilation does
        # not have yet. A header the user ships as a *path* is copied into the cache's
        # include folder and included by name; a header given as a full `#include` line
        # (a system header, say) is carried into the generated source verbatim.
        plan.library_header_lines:List[str] = []
        if plan.translator.device_library_headers:
            from ..device_library import include_lines_for

            plan.library_header_lines = include_lines_for(
                backend, plan.translator.device_library_headers, self._cache_dir(backend))
        if plan.library_header_lines:
            plan.library_includes:str = "\n".join(plan.library_header_lines)
            plan.include_lines = (f"{plan.include_lines}\n{plan.library_includes}" if plan.include_lines
                             else plan.library_includes)

        # Every top-level definition is separated by one blank line: helper from helper,
        # helper from kernel. A trailing blank on each piece makes the join produce exactly
        # one, and the final trim drops the one after the last definition.
        # One piece per rendered kernel: the merge identifies a definition by its first
        # signature line, so a multi-phase body joined into a single piece would replace
        # only the first phase's group and leave the file's later phases in place -- the
        # file then held those phases twice and the next compile refused them ("already
        # been defined").
        plan.pieces:List[str] = [piece.rstrip("\n") for piece in plan.device_prefix]
        plan.pieces.extend(source.rstrip("\n") for _, source in plan.rendered)

        plan.own_path:str = module_of({"source_file": plan.kernel_source_file})
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
        plan.ordered_foreign:List[str] = []
        for entry in plan.ordered_entries:
            plan.path:str = module_of(entry)
            if plan.path in plan.foreign_entries and plan.path not in plan.ordered_foreign:
                plan.ordered_foreign.append(plan.path)
        plan.foreign_includes:List[str] = [
            source_paths.include_line(plan.own_path, plan.path) for plan.path in plan.ordered_foreign]
        # A generated file is a *header* plus a list of definitions. The header names every
        # structure the file's own code uses and includes every other module it reaches;
        # the definitions are appended one at a time, so a module with several kernels keeps
        # all of them in one file. The header comes first because C needs a type before the
        # function that names it, and a callee before its caller.
        plan.own_header:List[str] = []
        if backend == "metal":
            # Metal's prelude: the `#include <metal_stdlib>` and the macros that give the
            # shared body its OpenCL index spellings (see kernel/metal_codegen.py). It
            # precedes the typedefs because the macros are the first thing the kernel body
            # needs -- and it must be first in the *file*, before any declaration.
            from ..metal_codegen import prelude
            plan.own_header.append(prelude())
        if plan.include_lines:
            # The fixed headers come BEFORE the typedef block: a structure member can name
            # `quatf` or `c64`, and that declaration has to be in scope when the structure
            # is declared, not after it.
            plan.own_header.append(plan.include_lines)
        if plan.typedefs:
            plan.own_header.append(plan.typedefs)
        if plan.foreign_includes:
            plan.own_header.append("\n".join(plan.foreign_includes))

        plan.written:Dict[str, str] = {}
        for plan.path, plan.entries in plan.foreign_entries.items():
            # A foreign module's file declares no structures of its own. The kernel's file
            # carries the whole typedef block ahead of the includes that reach these files,
            # and each declaration is guarded by its structure name, so the type is declared
            # once no matter how many files are compiled from it. Repeating the block here
            # would put every declaration in every file, which is a cycle of includes and a
            # duplicate definition in the one file two modules share.
            plan.foreign_header:List[str] = []
            if plan.include_lines:
                plan.foreign_header.append(plan.include_lines)
            # A foreign module may itself call into a third module: every structure is
            # declared by the top-level file, but a *function* it calls has to be reached
            # from its own file, and only by the files that call it.
            plan.callee_includes:List[str] = [
                source_paths.include_line(plan.path, other) for other in plan.ordered_foreign
                if other != plan.path and other in _called_modules(plan.path, plan.foreign_entries)]
            if plan.callee_includes:
                plan.foreign_header.append("\n".join(plan.callee_includes))
            plan.written[plan.path] = source_paths.append_source(
                plan.path, plan.foreign_header,
                [entry["code"] for entry in plan.entries if entry["code"]],
                plan.entries[0].get("source_file", "") or "",
                source_digest=plan.source_digest)

        plan.source_path:str = source_paths.append_source(
            plan.own_path, plan.own_header, [piece for piece in plan.pieces if piece],
            plan.kernel_source_file, source_digest=plan.source_digest)
