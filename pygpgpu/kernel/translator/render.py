from __future__ import annotations
import hashlib
from typing import (Dict, List, Optional, Set, Tuple)
from ..structs import dtype_for as structs_dtype_for
from .notation import (
    IMAGE,
    WORK_GROUP_VALUES,
    _CUDA_TYPE_NAMES,
    device_name,
)
from .predicates import (
    is_image,
    is_ptr,
)


class _RenderMixin:

    def _needs_prologue_parameters(self)->bool:
        """Whether this device function must receive Metal's work-item parameters.

        On OpenCL and CUDA the index builtins are *device intrinsics* -- `get_global_id`
        and `threadIdx` are symbols the runtime provides to every function. Metal has no
        such intrinsic: the prelude defines those names as macros over attribute-decorated
        **kernel parameters**, which exist only in the entry point's own scope.

        A helper that calls one therefore expands to a reference to a parameter it does
        not have. The generated MSL compiles into `'_mtl_thread_position' was not
        declared in this scope`, which names a name the user never wrote. Passing the six
        parameters down is what makes the helper legal, and it is only needed on Metal.
        """
        # `uses_global_id` is the translator's existing record of "the body names a
        # work-item index"; it is set where the index builtins are emitted.
        return self._metal_syntax and bool(self.uses_global_id)


    def _prologue_parameter_declarations(self)->List[str]:
        """`uint3 _mtl_thread_position [[thread_position_in_grid]]`, one per axis."""
        from ..metal_codegen import prologue_parameters
        return prologue_parameters()


    def _prologue_parameter_arguments(self)->List[str]:
        """The names a call site forwards for the six prologue parameters."""
        from ..metal_codegen import PROLOGUE_PARAMETERS
        return [name for _t, name, _a in PROLOGUE_PARAMETERS]


    def render_device(self, c_name:str,
                      shape_params:Optional[List[Tuple[str, int]]]=None)->str:
        """Render a device function.

        ``shape_params`` are the (array, dimension) extents this helper indexes. They travel
        as extra int parameters, because a pointer carries no extents and the helper's
        multi-dimensional subscripting needs them.
        """
        params:List[str] = []
        outer_names:Set[str] = set(self.outer_vars) | set(self.host_values) | set(self.host_scalars)
        for name, t in self.params:
            if name in outer_names:
                continue
            if is_ptr(t):
                # Every structure parameter is a pointer, and one whose members the body
                # never writes is `const` -- the same rule the kernel signature follows.
                from .. import structs as _structs_module  # noqa: PLC0415

                is_struct:bool = (_structs_module.dtype_for(t[1]) is not None
                                  if isinstance(t[1], str) else False)
                const:str = ("const " if is_struct and name not in self.member_writes
                             else "")
                params.append(f"{const}{self._device_type(t[1])}* {device_name(name)}")
            elif isinstance(t, str) and structs_dtype_for(t) is not None:
                # A structure value that reached a device parameter by value (a helper's
                # receiver before the pointer rule, or a caller's own spelling) travels as
                # a pointer too: one spelling for every structure.
                const = ("const " if name not in self.member_writes else "")
                params.append(f"{const}{self._device_type(t)}* {device_name(name)}")
            elif isinstance(t, tuple) and t and t[0] in WORK_GROUP_VALUES:
                # A work-group buffer handed to a helper -- a per-group slice or a
                # block-local array. It is a *pointer* into the group's own storage, in the
                # space that storage lives in -- Metal spells that `threadgroup`, OpenCL
                # `__local`, and CUDA/HIP need no qualifier. Rendering the tuple itself would
                # print the internal type marker into the C source, which no compiler
                # accepts.
                space:str = ""
                if self._metal_syntax:
                    space = "threadgroup "
                elif self._opencl_syntax:
                    space = "__local "
                element:str = self._device_type(t[1])
                params.append(f"{space}{element}* {device_name(name)}")
            elif isinstance(t, tuple) and len(t) == 3 and t[0] == IMAGE:
                # An image parameter reaching a helper. It is the backend's own image
                # type -- `__read_only image2d_t` on OpenCL, a texture/surface object on
                # CUDA -- so the same spelling the kernel signature uses applies here.
                # Falling through would print the internal marker (`('image', 4,
                # 'float')`) into the C source, which no compiler accepts.
                params.append(f"{self._image_param_type(name, t)} {device_name(name)}")
            else:
                params.append(f"{self._device_type(t)} {device_name(name)}")
        # A structure in the signature needs its typedef emitted too, and this helper's own
        # `private_structs` is what the caller collects.
        from .. import structs

        for candidate in [t[1] if is_ptr(t) else t for _, t in self.params]:
            dtype:object = structs.dtype_for(candidate) if isinstance(candidate, str) else None
            if dtype is not None and dtype not in self.private_structs:
                self.private_structs.append(dtype)
        if isinstance(self.return_type, str):
            dtype = structs.dtype_for(self.return_type)
            if dtype is not None and dtype not in self.private_structs:
                self.private_structs.append(dtype)
        for array_name, dim in (shape_params or []):
            params.append(f"const int {self._shape_var(array_name, dim)}")
        # Outer-scope parameters come last, matching the call site (see _emit_device_call),
        # which forwards them after the helper's own arguments and shape extents.
        params.extend(self._outer_param_declarations(device=True))
        if self._needs_prologue_parameters():
            # Metal's work-item numbers are parameters, so a helper that reads one has to
            # receive them from its caller (see _needs_prologue_parameters).
            params.extend(self._prologue_parameter_declarations())
        # A tuple return's output parameters, after everything else (the call site
        # appends and passes them in the same order -- see _emit_device_call).
        for ordinal, element_t in enumerate(self.tuple_return_types or []):
            params.append(f"{self._device_type(element_t)}* _tuple_out{ordinal}")
        ret:str = (self.return_type or "void")
        header:str = f"{self._device_qualifier()}{self._device_type(ret)} {c_name}({', '.join(params)})"
        body:str = self._render_body()
        # No typedef block here: the caller emits every structure type once, ahead of all
        # the device functions (see PyKernel._compile). Repeating it per helper would define
        # the same struct twice, which C++ rejects.
        return f"{header}\n{{\n{body}\n}}"


    def _outer_param_declarations(self, device:bool=False)->List[str]:
        """C declarations for the parameters the host fills in at launch time.

        This covers both the outer-scope variables and the objects the kernel body
        creates: neither belongs to the python argument list, and both are appended to the
        device signature after everything the work size is derived from.
        """
        declarations:List[str] = []
        for name, outer in self.outer_vars.items():
            if is_ptr(outer["type"]):
                qualifier:str = "" if device else self._buffer_qualifier()
                declarations.append(f"{qualifier}{self._device_type(outer['type'][1])}* {name}")
            elif outer["writable"]:
                element:str = (self._return_param_type(outer["type"]) if device
                               else self._kernel_scalar_type(outer["type"]))
                qualifier = "" if device else self._buffer_qualifier()
                declarations.append(f"{qualifier}{element}* {name}")
            else:
                declarations.append(f"{self._kernel_scalar_type(outer['type'])} {name}")
        for name, entry in self.host_values.items():
            declarations.append(f"{self._host_value_declaration(name, entry, device)}")
        # The plain values a host-region statement bound: they are filled in at launch
        # time exactly like a creation, so they are declared with them. A scalar is a
        # plain parameter; an array the host built is a buffer, the same spelling a
        # creation of that shape gets.
        for name, t in self.host_scalars.items():
            if name in self.outer_vars:
                # The outer scope already declared it (an enclosing `nonlocal` slot): one
                # parameter, and it is the slot, not a second by-value copy.
                continue
            if is_ptr(t):
                qualifier = "" if device else self._buffer_qualifier()
                declarations.append(f"{qualifier}{self._device_type(t[1])}* {device_name(name)}")
            else:
                declarations.append(f"{self._kernel_scalar_type(t)} {device_name(name)}")
        return declarations


    def _host_value_declaration(self, name:str, entry:Dict[str, object], device:bool)->str:
        """The parameter declaration for one host-created object."""
        t:object = entry["type"]
        if is_ptr(t):
            qualifier:str = "" if device else self._buffer_qualifier()
            return f"{qualifier}{self._device_type(t[1])}* {device_name(name)}"
        return f"{self._kernel_scalar_type(t)} {device_name(name)}"


    def _device_type(self, t:str)->str:
        """The backend's spelling of a scalar type name.

        The translator's internal names follow OpenCL (uchar, ushort, uint, ulong). CUDA
        C does not define those four, so the CUDA path -- which HIP shares -- writes the
        full C form instead.
        """
        if self._opencl_syntax:
            return t
        return _CUDA_TYPE_NAMES.get(t, t)


    def _kernel_scalar_type(self, t:object)->str:
        """A kernel parameter's C type.

        `bool` is not portable as a kernel parameter: OpenCL C refuses a `_Bool` kernel
        argument outright, and the CUDA/HIP argument binding has no bool entry. It travels
        as a byte instead, which every backend accepts; the device reads 0/1.
        """
        if t == "bool":
            return self._device_type("uchar")
        return self._device_type(t)


    def parameter_declarations(self)->List[str]:
        """The kernel's parameter list, exactly as the generated signature spells it.

        Split out of `render_kernel` because the kernel's C name has to cover it: the name's
        hash is built from the *argument types* the caller passed, which do not include the
        extent parameters the translator derives, nor the outer-scope values, nor the shared
        buffers. A body whose derived signature changed therefore kept its old name, and the
        file then held two declarations of that name -- "more than one instance of overloaded
        function". Hashing the rendered list makes a changed signature a changed name.
        """
        # Outer-scope parameters are held in self.params so the host can index launch
        # arguments by name, but they are *declared* last: the host appends them after the
        # work-size/sampler/local-size arguments it derives (see PyKernel._invoke_inner).
        outer_names:Set[str] = set(self.outer_vars) | set(self.host_values) | set(self.host_scalars)
        params:List[str] = []
        for name, t in self.params:
            if name in outer_names:
                continue
            if is_ptr(t):
                # A structure parameter is a pointer, and one the body never writes is
                # `const`: it says the member reads are all the callee does, which is the
                # information a reader of the generated source needs.
                from .. import structs as _structs_module  # noqa: PLC0415

                is_struct:bool = (_structs_module.dtype_for(t[1]) is not None
                                  if isinstance(t[1], str) else False)
                const:str = ("const " if is_struct and name not in self.member_writes
                             else "")
                qualifier = self._buffer_qualifier()
                params.append(
                    f"{qualifier}{const}{self._kernel_scalar_type(t[1])}* {device_name(name)}")
            elif is_image(t):
                params.append(f"{self._image_param_type(name, t)} {device_name(name)}")
            else:
                params.append(f"{self._kernel_scalar_type(t)} {device_name(name)}")
        for array_name, dim in self.shapes_used:
            params.append(f"const int {self._shape_var(array_name, dim)}")
        # OpenCL sampling takes the sampler as a kernel argument; one per sampled
        # image, right after that image. The host binds a matching sampler per image,
        # so the Python-visible signature stays (image) only. Metal has no sampler
        # argument at all -- a texture carries its sampler state -- so this stays
        # OpenCL's alone rather than following `_opencl_syntax`.
        if self.backend == "opencl":
            for image_name in sorted(self.required_image_samplers):
                params.append(f"const sampler_t {self._sampler_name(image_name)}")
        for lc_name, lc_type, _ in self.local_buffers:
            if self._metal_syntax:
                # MSL spells the work-group address space `threadgroup`, and a
                # `threadgroup` buffer is a kernel parameter there too.
                params.append(f"threadgroup {self._device_type(lc_type)}* {lc_name}")
            elif self._opencl_syntax:
                params.append(f"__local {self._device_type(lc_type)}* {lc_name}")

        # Outer-scope parameters after everything the launch derives the work size from,
        # matching the order the host appends them in (see PyKernel._invoke_inner).
        params.extend(self._outer_param_declarations(device=False))

        # Return slots come last, after every parameter the work size and samplers derive
        # from, so appending them at launch time cannot disturb those positions.
        params.extend(self.render_return_params())

        # Metal's work-item numbers arrive as attribute-decorated parameters, not as
        # `get_global_id()` calls (see kernel/metal_codegen.py). They are declared after
        # every parameter the host binds, because the host never binds them: the dispatch
        # fills the attributes in. The names are the ones the prelude's macros expand
        # onto, so a body keeps its OpenCL spelling.
        if self._metal_syntax:
            from ..metal_codegen import prologue_parameters
            params.extend(prologue_parameters())
        return params


    def signature_digest(self)->str:
        """A digest of what this kernel's device signature declares.

        The C name appends it, so a body whose derived parameters changed is a different
        name -- and the generated file then holds one declaration instead of two that
        collide.
        """
        return hashlib.md5(";".join(self.parameter_declarations()).encode("utf-8")).hexdigest()[:8]


    def render_kernel(self, c_name:str)->str:
        params:List[str] = self.parameter_declarations()

        if self._metal_syntax:
            from ..metal_codegen import kernel_prefix
            prefix:str = kernel_prefix() + " "   # "kernel void ", the return type included
        else:
            prefix = "__kernel " if self._opencl_syntax else "__global__ "
        # Metal's prefix carries the return type (`kernel void`); every other backend
        # spells the qualifier alone, so the `void` is appended only for them.
        return_type:str = "" if self._metal_syntax else "void "
        header:str = f"{prefix}{return_type}{c_name}({', '.join(params)})"
        body_lines:List[str] = []
        if self.backend == "cuda":
            for lc_name, lc_type, lc_count in self.local_buffers:
                # `_device_type` because CUDA C has no `uint`: the shared array has to be
                # declared in the backend's spelling, not the translator's internal one.
                # One indent level, like a body statement: these lines are the kernel's
                # first declarations, not part of the signature.
                body_lines.append(
                    f"    __shared__ {self._device_type(lc_type)} {lc_name}[{lc_count}];")
        body_lines.extend(self.lines)
        # A blank line between the shared-memory declarations and the body reads as a
        # separate preamble; the renderer collapses and trims whatever follows.
        body_lines.append("")
        body:str = self._render_body(body_lines)
        # The structure types are emitted once by the caller, ahead of the whole source (see
        # PyKernel._compile): a helper may need them too, and two copies of the same struct
        # is an error in C++.
        # The body already ends with the separator rule applied: two braces that close a
        # block and its parent sit together, with no blank line between them.
        return f"{header}\n{{\n{body}\n}}"


    def render_kernels(self, c_name:str)->List[Tuple[str, str]]:
        """Every kernel this translation produces, as (name, rendered source).

        A body with one magic loop renders one kernel. A body with several renders one per
        phase, each named after the phase so the host can launch them in order; they share
        the source, so a phase's helpers and typedefs are emitted once alongside the first.
        """
        if not self.phase_translators:
            return [(c_name, self.render_kernel(c_name))]
        rendered:List[Tuple[str, str]] = []
        for index, phase in enumerate(self.phase_translators):
            phase_name:str = f"{c_name}_p{index}"
            rendered.append((phase_name, phase.render_kernel(phase_name)))
        return rendered
