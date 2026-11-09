# The constructor of the translator: the state every translation starts from.
from __future__ import annotations

import ast
import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .notation import (
    DeviceType,
    HostValueEntry,
    OuterVarEntry,
    ReturnLayoutEntry,
    ReturnPosition,
    Translated,
)
from .source_text import _parse_kernel_source, _source_first_line, _trimmed_source

if TYPE_CHECKING:
    from . import FunctionTranslator


class _InitMixin:

    def __init__(self, func:Callable, param_types:List[object], backend:str,
                     resolve_global:Callable[[str], Any], device_cache:Dict[object, object],
                     is_kernel:bool=False, self_param:str="",
                     self_attr_types:Optional[Dict[str, object]]=None,
                     self_owner:Optional[type]=None, launch_backend:Optional[str]=None,
                     host_creation_values:Optional[Dict[str, object]]=None,
                     host_executed_lines:Optional[List[int]]=None,
                     self_struct_members:Optional[Dict[str, str]]=None,
                     self_struct_type:object=None,
                     struct_arg_classes:Optional[Dict[str, type]]=None,
                     receiver_as_argument:bool=False,
                     type_map:Optional[Dict[str, str]]=None,
                     argument_values:Optional[List[object]]=None):
            self.func:Callable = func
            self.filename:str = getattr(getattr(func, "__code__", None), "co_filename", "<kernel>")
            self.backend:str = backend
            # The backend the kernel is actually launched on. hip reuses cuda code generation, so
            # backend is "cuda" for a hip kernel; decisions that depend on the runtime rather than
            # on the generated syntax (for example whether device printf is safe) must use this.
            self.launch_backend:str = (launch_backend or backend)
            # The code-generation shape, as opposed to the runtime: "opencl" and "metal" share
            # the C-family emission (kernels, pointers into buffers, atomics, the same vector
            # types), so every site that asks "is this the OpenCL-looking syntax" tests this
            # rather than the backend name -- and a site that needs Metal's own spelling tests
            # `metal_syntax`. Two properties rather than a name comparison keep the two
            # questions apart: "does this backend spell a kernel `__kernel void`" is not the
            # same question as "is this backend Metal's attributes".
            self._opencl_syntax:bool = backend in ("opencl", "metal")
            self._metal_syntax:bool = backend == "metal"
            self._c_syntax:bool = backend in ("cuda", "hip")
            self.resolve_global:Callable[[str], Any] = resolve_global
            self.device_cache:Dict[object, object] = device_cache
            self.is_kernel:bool = is_kernel
            # The argument types the kernel was specialized for; kept so a phase translator can
            # be built with the same signature (see _translate_phase_kernels).
            self.param_types:List[object] = list(param_types)
            # Kernels launched from this kernel's device code (`inner[(n,)](args)`). Each entry is
            # one rendered `__global__` function, emitted ahead of the outer kernel because C
            # requires a callee before its caller.
            self.nested_kernels:List[str] = []
            # callee -> the name its definition was given, so two launches share one definition.
            self.nested_seen:Dict[Callable, str] = {}
            # Device helpers a nested kernel needs, emitted once for the whole source.
            self.nested_helpers:List[str] = []
            # Device helpers this kernel's own body needs, emitted once for the whole source.
            # A built-in that a backend spells with a helper (the round-toward-zero forms, which
            # HIP has no intrinsic for) contributes one of these; C requires the definition
            # before the kernel that calls it.
            self.device_helpers:List[str] = []
            # Set when this function's body reads a work-item index (`get_global_id`, or a
            # `blockIdx`/`threadIdx` name that resolves to one). Metal needs it to decide
            # whether a helper has to receive the six prologue parameters; every other
            # backend ignores the flag (see `_needs_prologue_parameters`).
            self._uses_index_builtins:bool = False
            # One entry per magic loop when a body has several; each becomes its own kernel.
            self.phases:List[List[ast.stmt]] = []
            # Locals initialized above the magic loop and written inside it: one location for the
            # whole launch, carried as a one-element buffer (see _bind_carried_scalars).
            self.carried_scalars:List[str] = []
            # Names a statement between two magic loops computed on the host: each travels to the
            # pass that reads it as its own parameter (see _translate_phase_kernels).
            self.gap_names:List[str] = []
            # The statements of the gap that follows this phase's loop (set by
            # _translate_phase_kernels before the body is visited).
            self.phase_gap:List[ast.stmt] = []
            # The statements after the last magic loop of a single-phase body: host code, run
            # once after the launch (see _translate_magic_loop and PyKernel._run_host_region).
            self.body_tail:List[ast.stmt] = []
            # The whole-body carried set the main translator computed, so a phase that writes
            # one of these names keeps the shared slot (see _bind_carried_scalars).
            self.phase_carried_hint:List[str] = []
            # By-value arguments the body returns, promoted to pointers so the return reuses the
            # parameter instead of allocating a slot for a copy (see _promote_returned_parameters).
            self.promoted_parameters:Set[str] = set()
            # The promoted parameters a return site lands in, so no slot is allocated for them.
            self.promoted_returns:Set[str] = set()
            # Local names that are only another name for an array (`src = a`). The device reads
            # the same pointer, so the alias asks for its target's extents (see _shape_var).
            self._array_pointer_sources:Dict[str, str] = {}
            # A local bound to a slice (`part = a[lo:hi]`) keeps the length here: it is a
            # pointer into the same buffer plus an offset, and `part.shape[0]` answers this.
            self._array_slice_extents:Dict[str, str] = {}
            # The array parameters a local was sliced from (`part = a[lo:hi]`). A slice is a
            # pointer at an offset, which matches python's view only on a 1-D array, so the
            # launch checks the rank of each of these (see PyKernel._compile's sliced_arrays).
            self.sliced_arrays:List[str] = []
            # `enumerate([1.0, 2.0, ...])`: a literal list is a compile-time table, so its
            # elements are recorded here and the dispatch reads them (see the enumerate branch).
            self._enumerate_constants:List[float] = []
            # A name bound to a literal list that a loop enumerates (`weights = [1.0, 2.0]`):
            # the loop reads the elements themselves, so the binding is not device code.
            self._constant_list_names:Set[str] = set()
            self._constant_lists:Dict[str, List[float]] = {}
            self.phase_translators:List["FunctionTranslator"] = []
            self.self_param:str = self_param
            self.self_attr_types:Dict[str, object] = dict(self_attr_types or {})
            # The class the translated method belongs to, for deciding whether `self.name`
            # is a data attribute, an instance method, or a staticmethod. None for the
            # decoration-time body scan, where the class does not exist yet.
            self.self_owner:Optional[type] = self_owner
            # The class each STRUCTURE argument was actually built from. One dtype can stand
            # for several classes -- two classes with one layout share it -- so the dtype alone
            # cannot say whose method `p.total` is. The launch knows: the caller handed over an
            # instance, and its type is recorded here under the parameter's name.
            self.struct_arg_classes:Dict[str, type] = dict(struct_arg_classes or {})
            # The receiver travels as one structure: member name -> the attribute it holds.
            # A member read of `self.<attr>` becomes a field read of the struct parameter, and
            # an ndarray attribute stays a separate `_self_<attr>` argument (the read/write
            # analysis and the shape plumbing need it apart).
            self.self_struct_members:Dict[str, str] = dict(self_struct_members or {})
            self.self_struct_type:object = self_struct_type
            # True when this function's receiver is its own first positional parameter
            # (`def at(self, t)` called as `ray.at(t)`), rather than a field-carrying struct
            # the caller packs separately (see _emit_device_call).
            self.receiver_as_argument:bool = receiver_as_argument
            # Objects the host already built for statements above the magic loop, keyed by the
            # name the body assigns them to (see _host_created_assignment).
            self.host_creation_values:Dict[str, object] = dict(host_creation_values or {})
            # The values every *host region* bound, created or not, keyed the same way, and the
            # line numbers of the statements the interpreter actually ran. A statement in that
            # set is host work -- it runs once, on the host, before the launch -- so the device
            # must not run it too (see _host_statement_lines and _bind_host_statement).
            self.host_region_values:Dict[str, object] = dict(host_creation_values or {})
            # The caller's actual arguments, in positional order, so a compile-time fact the
            # body needs can read the *value* a parameter carries this launch. Only
            # `zeros_per_group`'s shape reads it today: a shared buffer's size is baked into
            # the generated source, so it specializes the kernel the way a type does, and
            # the value is part of the signature digest that hashes the C name.
            self.argument_values:List[object] = list(argument_values or [])
            # The plain values a host-region statement bound, by name -> device type (see
            # _bind_host_statement). They are parameters like any other, but they are not
            # *creations*, so the launch must not treat them as objects.
            self.host_scalars:Dict[str, DeviceType] = {}
            self.host_statement_lines:Set[int] = set(host_executed_lines or [])
            # The width a python `int`/`float` travels as (device type names, already resolved).
            # Every type query the translator makes on a *host* value -- a creation above the
            # loop, a constant it lifts -- goes through it, so a kernel and the objects it builds
            # agree on the width.
            self.type_map:Dict[str, str] = dict(type_map or {})

            try:
                # The dedented copy starts at the decorator line, so remember where that was in
                # the user's file: error line numbers are reported against the file, not the copy.
                self.first_line:int = _source_first_line(func)
                self.source_text:str = _trimmed_source(func)
                # A translator *owns* its tree: a multi-pass body hands each of its phases a
                # slice by assigning `phase.tree.body`, and a phase translator parses the same
                # text as the main one. `_parse_kernel_source` answers a cached tree, so the
                # copy is what keeps one phase's transplant from rewriting the main
                # translator's body (and every later phase's). It is paid once per compile,
                # not once per launch.
                tree:ast.Module = copy.deepcopy(_parse_kernel_source(self.source_text))
                if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
                    raise SyntaxError("decorated object must be a plain function") from None
                self.tree:ast.FunctionDef = tree.body[0]
            except SyntaxError as e:
                raise SyntaxError(f"cannot parse function source: {e}") from None

            self.params:List[Translated] = []
            self.shapes_used:List[Tuple[str, int]] = []
            # Structure parameters whose members the body assigns: a parameter in this set is
            # emitted without `const` (see render_kernel / render_device).
            self.member_writes:Set[str] = set()
            self.gid_param:Optional[str] = None
            self.work_size_dims:List[Tuple[str, int]] = []
            self.work_size_members:List[str] = []
            # One (array, dimension) source per dimension of the ndindex magic loop, in the
            # order the loop unpacks them (see _ndindex_extents). None for the other forms.
            self._nd_loop_extents:Optional[List[Tuple[str, int]]] = None
            # The array named by np.ndindex(arr.shape), whose rank is the loop's unpacking.
            self._nd_shape_holder:str = ""
            # name -> how many indices the body subscripts that array with (see
            # _record_subscript_indices); checked against the real shape at launch.
            self.subscript_indices:Dict[str, int] = {}
            # name -> the line of the widest such subscript, so the launch can blame that
            # statement rather than the loop when the count exceeds the array's rank.
            self.subscript_lines:Dict[str, int] = {}
            # name -> the column that subscript starts at, so the caret lands on the indexed
            # expression (`a[i, i]`), the way numpy's own IndexError points at it.
            self.subscript_columns:Dict[str, int] = {}
            # name -> the subscript as the user wrote it (`image[i, j, 2]`). The launch hands
            # it to the frame stand-in, which re-runs that very expression against an array of
            # the wrong rank so CPython draws its own caret instead of a synthetic one.
            self.subscript_texts:Dict[str, str] = {}
            # The magic loop's index variables, in unpack order. A subscript formed from
            # exactly these names addresses a different element per work item, which is what
            # makes a plain read-modify-write safe (see _augassign_is_race_free).
            self.work_item_indices:List[str] = []
            # Structure dtypes a private array created in the body mentions: their typedefs
            # have to be emitted even though they are not parameters.
            self.private_structs:List[object] = []
            # Dtypes synthesized from a plain class the body constructs (`ray = Ray(...)` on a
            # class that is not a ctypes.Structure). Their members are what the body reads, so a
            # construction may hand over more than that and the unread arguments are dropped --
            # a declared structure, whose layout is fixed, still refuses them.
            self.plain_class_dtypes:Set[np.dtype] = set()
            # Generated quaternion device functions, keyed by (name, body) so one definition is
            # emitted however many call sites ask for it.
            self.quaternion_helpers:Dict[str, str] = {}
            # The quaternion blocks a generated source needs, as the `QUAT_NEED_*` macro names. One
            # header serves both widths; only the blocks in use have to be compiled.
            self.quaternion_needs:Set[str] = set()
            # The complex blocks this source needs, as the `COMPLEX_NEED_*` macro names -- the
            # same contract as the quaternion widths: one header, and only the widths in use are
            # compiled (a `double` block does not compile where fp64 is absent).
            self.complex_needs:Set[str] = set()
            # Set when the body reads or writes a complex value at all, which is what decides
            # whether the generated source includes the complex header.
            self.uses_complex:bool = False
            # The headers a registered device-side library function asked for: what the registry
            # spells for them (a path, or a full `#include` line). The compile step turns them
            # into the include lines the generated source carries.
            self.device_library_headers:List[str] = []
            # The array whose whole shape a `np.ndindex(arr.shape)` loop iterates, when that is
            # how the domain was spelled: the launch checks its rank against the loop's names.
            self.nd_whole_shape_param:str = ""
            # The array whose shape declares the iteration domain, recorded from
            # `@kernel(domain=...)`; this is the gid-style counterpart of a magic loop's implied
            # domain. The rank comes from that array's runtime shape (a pointer type carries no
            # rank), so PyKernel expands it at call time.
            self.declared_work_arrays:List[str] = []
            # Set when the body calls gpu.get_global_id()/... , i.e. it dispatches on the launch
            # index and therefore needs an iteration domain from somewhere.
            self.uses_global_id:bool = False
            # Set once a magic loop is detected: it fixes the iteration domain, so a
            # `@kernel(domain=...)` declaration would be redundant.
            self.magic_loop_present:bool = False
            # The line of that loop (in the parsed copy's numbering), so a launch-time rank
            # check can blame the `for` statement instead of the whole kernel.
            self.magic_loop_line:int = 0
            # Image parameters that an OpenCL kernel samples. Each one needs a sampler
            # constant emitted next to the kernel, and the host side builds one sampler
            # per image, so this set drives that generation.
            self.required_image_samplers:Set[str] = set()
            # Image parameters a CUDA kernel stores into: those need a surface handle,
            # which is a separate device object from the texture.
            self.required_image_surfaces:Set[str] = set()
            self.atomic_targets:Set[str] = set()
            self._tmp_counter:int = 0
            self.local_buffers:List[Tuple[str, str, int]] = []
            self.return_type:Optional[object] = None
            # A device helper's `return a, b` (see _visit_return): the element types, in
            # order. The call site appends one `T*` output parameter per element and passes
            # the destinations it unpacks into; the helper stores through them.
            self.tuple_return_types:Optional[List[object]] = None
            # The element types read for the tuple call currently being unpacked (see
            # _tuple_call_element_count, which stashes them for _emit_tuple_call_unpack).
            self._pending_tuple_types:Optional[List[object]] = None
            # (line index, name, initial type) for every local declaration `_visit_assign`
            # emitted, so the renderer can rewrite one whose variable a later assignment
            # widened (float -> double; see _define).
            self._declaration_sites:List[Tuple[int, str, str]] = []
            # Locals whose declaration moves to the top of the function because their first
            # assignment is inside a branch or loop; python scopes a name to the whole
            # function, so a read after the block needs the declaration there.
            self._hoisted_declarations:List[Tuple[str, str, str]] = []
            self._hoisted_names:Set[str] = set()
            # A local bound to an array extent above the magic loop (`rows: int =
            # out.shape[0]`): the name resolves to (array, dim), which is what an
            # `np.ndindex(rows, ...)` bound needs to become the launch's extent parameter.
            self._shape_aliases:Dict[str, Tuple[str, int]] = {}
            # An ndenumerate index held as one name (`for index, v in ...`): the name stands
            # for the coordinate tuple, and a constant-ordinal subscript reads the
            # decomposition the head emitted (python's order, slowest first).
            self._ndenumerate_index_names:Dict[str, List[str]] = {}
            # Kernel return values. A kernel runs once per work item, so it cannot have a C
            # return value; every returned position becomes a device parameter instead (a
            # one-element buffer for a value, the array parameter's own name for an array) and
            # the first work item to reach a `return` claims the whole result.
            #
            # return_layout: one entry per value slot ({"ctype", "type"}), shared by every
            #   return site whose value has that C type.
            # return_sites: per `return` statement, its positions as the translator saw them.
            # _site_slot_map: per site, (position ordinal, slot index) pairs it writes.
            self.return_layout:List[ReturnLayoutEntry] = []
            self.return_sites:List[List[ReturnPosition]] = []
            self._site_slot_map:Dict[int, List[Tuple[int, int]]] = {}
            # True when statements currently being visited run once per work item after the
            # magic loop (see _visit_region and _visit_kernel_return).
            self.after_magic_loop:bool = False
            # True when statements currently being visited are inside (or after) the magic
            # loop, where building data would happen once per work item (see _visit_assign).
            self.creation_below_loop:bool = False
            # The conditions of the enclosing `if` blocks, innermost last. A store to a
            # launch-wide location is safe when a guard decides the same way for every work
            # item (`if i == 0:`); see `_store_is_guarded`.
            self._guard_conditions:List[ast.expr] = []
            # True while the statements being visited are the magic loop's own body -- one
            # execution per work item, as opposed to `after_magic_loop`, which means after the
            # dispatch. A shared store is racy in the first case and refused (see
            # `_named_outer_target`), and the two flags are deliberately separate because a
            # statement inside the loop is not "after" it.
            self.in_work_item_body:bool = False
            # True when any site returns a value; then the kernel needs the claim and site
            # parameters even if some other site is a bare `return`.
            self.has_returns:bool = False
            # True when a `return` actually reaches the generated device code. A body whose only
            # return sits after the magic loop hands its value back through the host tail (see
            # PyKernel._run_host_region), so it never claims, and the claim/site words it would
            # otherwise declare are two wasted uploads and two never-read parameters.
            self._claims_on_device:bool = False
            # How many real C loops enclose the statement being visited. Only a nested serial
            # `for`/`while` that is emitted as a loop counts: the magic loop itself does not,
            # because it becomes the work-item dispatch rather than a loop in the generated
            # source (see _visit_work_item_body). `break` and `continue` are meaningful only
            # while this is non-zero, which is what decides how they are translated.
            self.loop_depth:int = 0
            # Outer-scope variables: a name the body uses that is neither a parameter nor a
            # local becomes a device parameter instead (see _bind_outer_variables). Each entry
            # records how the host reaches the name ("closure" through a cell, "global" through
            # func.__globals__), its device type, and whether the body declares it writable
            # (`nonlocal`/`global`) -- which decides between `T* name` + `*name` and a plain `T`.
            self.outer_vars:Dict[str, OuterVarEntry] = {}
            # Values the host creates for the kernel: an assignment above the magic loop whose
            # right-hand side only builds data (np.zeros, np.full, ...) is executed by the
            # interpreter, not translated, and the object travels as a parameter (see
            # _host_created_assignment).
            self.host_values:Dict[str, HostValueEntry] = {}
            self.return_type:Optional[object] = None
            self.lines:List[str] = []
            self._indent:int = 1
            self.scopes:List[Dict[str, DeviceType]] = []

            args:List[ast.arg] = self.tree.args.args
            if self.tree.args.vararg or self.tree.args.kwarg or self.tree.args.kwonlyargs:
                raise SyntaxError("*args/**kwargs/keyword-only args are not supported", self.tree.args) from None
            # Default values are accepted: a kernel argument is always supplied by the
            # caller, so a default is a convenience for reading the signature rather than
            # something the device code has to represent.
            receiver_type:object = None
            if self_param and args and args[0].arg == self_param:
                if self.receiver_as_argument:
                    # The receiver travels as this function's own first argument: take its
                    # type off the front and name it before the remaining parameters.
                    receiver_type = param_types[0] if param_types else None
                    param_types = list(param_types[1:])
                args = args[1:]
            # `param_types` holds the positional arguments, then the receiver struct, then one
            # type per ndarray attribute (see PyKernel._invoke_inner and _emit_device_call).
            expected:int = len(args) + len(self.self_attr_types) + (
                1 if self.self_struct_members else 0)
            if expected != len(param_types):
                raise SyntaxError(f"{func.__name__}() takes {expected} device arguments but {len(param_types)} types were given") from None

            # The receiver is a method's first parameter, the way python spells one: `self` is
            # the receiver, so the device signature reads `Class* self,` before the caller's
            # own arguments. `self.params` *is* the signature order, so the receiver is
            # appended before the arguments rather than after them. The slot it occupies in
            # `param_types` stays where the host packs it (after the positional arguments), so
            # the type is taken from there and the two lists keep their own orders.
            scope:Dict[str, DeviceType] = {}
            if self.self_struct_members:
                struct_type:object = (param_types[len(args)] if len(param_types) > len(args)
                                   else self.self_struct_type)
                if self.self_struct_type is None:
                    self.self_struct_type = struct_type
                self.params.append((self.self_param, struct_type))
            if receiver_type is not None:
                # The receiver is not a `_self_` field set: it is one structure pointer,
                # spelled by the function's own first parameter name.
                scope[self.self_param] = receiver_type
                self.params.append((self.self_param, receiver_type))
            for arg, t in zip(args, param_types):
                # Annotations are accepted but ignored: the device type comes from the
                # actual argument at call time (see infer_ctype), because one kernel is
                # specialized per argument-type signature. Keeping them lets the same
                # function be read by editors and type checkers.
                scope[arg.arg] = t
                self.params.append((arg.arg, t))
            for attr, t in self.self_attr_types.items():
                if attr in self.self_struct_members:
                    continue                  # carried inside the receiver struct
                pname:str = f"_self_{attr}"
                scope[pname] = t
                self.params.append((pname, t))
            self.scopes.append(scope)
