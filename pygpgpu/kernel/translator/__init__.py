from __future__ import annotations
import ast
import builtins
import enum
import hashlib
import inspect
import math
import os
import re
import textwrap
import types
from typing import (Any, Callable, Dict, List, NoReturn, Optional, Sequence,
                    Set, Tuple, Union, cast)
import numpy as np
from ..builtin_registry import builtin_name_of
from ..device_library import library_function_name, library_functions
from ..structs import dtype_for as structs_dtype_for, member_types as structs_member_types
from .. import complex as complex_support
from ..device_builtins import (
    ATOMIC_BUILTINS,
    CUDA_ONLY_BUILTINS,
    CUDA_ONLY_CONVERSIONS,
    GENERIC_BUILTINS,
    HIP_ROUND_TOWARD_ZERO_HELPERS,
    HIP_ROUND_TOWARD_ZERO_SOURCES,
    INDEX_BUILTINS,
    INDEX_VECTOR_AXES_CUDA,
    INDEX_VECTOR_AXES_OPENCL,
    INDEX_VECTOR_BUILTINS,
    INDEX_VECTOR_OPENCL,
    MATH_BUILTINS,
    SHARED_MATH_BUILTINS,
    SHARED_MATH_TEMPLATES,
    emit_atomic,
    emit_barrier,
    emit_index,
    float_atomic_add_source,
    python_floor_name,
    python_floor_source,
)
from .source_text import (
    _copy_ast,
    _get_attr,
    _has_device_return,
    _normalize_ast_legacy_nodes,
    _parse_kernel_source,
    _returns_a_value,
    _source_first_line,
    _trimmed_source,
    get_dedented_source,
)
from .errors import (
    _position_error,
    _syntax_error,
    _unpack_count_message,
    _unpacking_error,
)
from .type_map import (
    COMPLEX_TYPE_NAMES,
    CTYPES_SCALAR_TYPES,
    DTYPE_MAP,
    FLOAT_MAPPING_TARGETS,
    FLOAT_TYPES,
    INTEGER_MAPPING_TARGETS,
    INT_TYPES,
    NARROWED_SCALARS,
    SCALAR_TYPES,
    TYPE_MAP_KEYS,
    UNSIGNED_INTEGER_TYPES,
    _COMPLEX_WIDTH_PER_REAL,
    _INTEGER_RANGES,
    _check_integer_fits,
    _ctypes_scalar_types,
    _mapping_refusal,
    check_integer_range,
    ctypes_device_type_name,
    python_complex_type,
    resolve_device_type,
    validate_type_map,
)
from .notation import (
    DeviceType,
    HostValueEntry,
    IMAGE,
    OuterVarEntry,
    PTR,
    RESERVED_DEVICE_NAMES,
    RESERVED_PREFIX,
    ReturnLayoutEntry,
    ReturnPosition,
    ReturnSiteLayout,
    ReturnSitePosition,
    SHARED_SLICE,
    Translated,
    _CUDA_TYPE_NAMES,
    device_name,
)
from .receivers import (
    BUILTIN_METHODS,
    CALL_METHOD,
    CONVERSION_METHODS,
    MAGIC_COMPARE_REFLECTED,
    MAGIC_INPLACE,
    MAGIC_METHODS,
    MAGIC_REFLECTED,
    _MISSING,
    _class_attr,
    _collect_self_attrs,
    _enum_member_value,
    _helper_globals,
    _magic_method_of,
    _operator_name,
    _property_accessor,
    _property_carrier,
    _property_of,
    _receiver_param_of,
    _self_member_call_kind,
    scan_self_attrs,
    self_attr_kind,
    structs_name,
    writes_receiver,
)
from .vectors import (
    _C_FIELD_TYPES,
    _VEC_COLOR_ALIASES,
    _c_scalar_name,
    _vec_component_name,
    _vec_components,
    _vec_field_names,
    _vec_registry,
    is_vec,
)
from .builtins import (
    COMPLEX_ABS_FUNCTIONS,
    COMPLEX_BUILTIN_FUNCTIONS,
    COMPLEX_BUILTIN_REAL_RESULTS,
    COMPLEX_BUILTIN_REFUSALS,
    COMPLEX_DEVICE_FUNCTIONS,
    COMPLEX_UNSUPPORTED_FUNCTIONS,
    KERNEL_TRANSLATING_NAME,
    KERNEL_TRANSLATING_VALUE,
    PRINT_INT_CASTS,
    PRINT_INT_FORMATS,
    PRINT_MAX_LINES,
    PRINT_REAL_BIG,
    PRINT_REAL_DIGITS,
    PRINT_REAL_PLAIN_MAX,
    PRINT_REAL_SCI_HI,
    PRINT_REAL_SCI_LO,
    PRINT_REAL_SPECIAL_COUNT,
    _MATH_CONSTS,
    _MATH_FUNCS,
    _math_device_name,
    _translation_branch,
    _translation_truth,
)
from .predicates import (
    SURFACE_HANDLE_TYPES,
    TEXTURE_HANDLE_TYPES,
    _handle_device_type,
    _is_device_handle,
    image_channels,
    image_scalar,
    infer_ctype,
    is_aggregate,
    is_complex,
    is_float,
    is_image,
    is_int,
    is_ptr,
    promote,
)
from .walk import (
    _BUILTIN_NAMES,
    _HOST_CREATION_FUNCTIONS,
    _NUMPY_DEVICE_TYPES,
    _UNRESOLVED,
    _assigned_name_of,
    _constant_int_of,
    _creation_assignments,
    _creation_target_of,
    _declared_scope_names,
    _is_creation_expression as _module_is_creation_expression,
    _is_creation_shape,
    _is_host_evaluable,
    _is_magic_loop_stmt,
    _is_shape_access,
    _magic_loop_end,
    _magic_loop_index,
    _outer_scope_names,
    _outside_loops,
    short_hash,
    type_sig,
)
from .emit import _EmitMixin
from .naming import _NamingMixin
from .phases import _PhasesMixin
from .magic_loop import _MagicLoopMixin
from .host_regions import _HostRegionsMixin
from .outer_vars import _OuterVarsMixin
from .statements import _StatementsMixin
from .returns import _ReturnsMixin
from .assign import _AssignMixin
from .augassign import _AugassignMixin
from .targets import _TargetsMixin
from .tuples import _TuplesMixin
from .print_emit import _PrintEmitMixin
from .expressions import _ExpressionsMixin
from .operators import _OperatorsMixin
from .subscripts import _SubscriptsMixin
from .images import _ImagesMixin
from .attributes import _AttributesMixin
from .complex_values import _ComplexValuesMixin
from .quaternions import _QuaternionsMixin
from .magic_methods import _MagicMethodsMixin
from .vector_ops import _VectorOpsMixin
from .structures import _StructuresMixin
from .calls import _CallsMixin
from .builtin_emit import _BuiltinEmitMixin
from .numpy_emit import _NumpyEmitMixin
from .device_call import _DeviceCallMixin
from .render import _RenderMixin
from .builtin_python import _BuiltinPythonMixin
from .nd_index import _NdIndexMixin
from .private_array import _PrivateArrayMixin
from .nested_launch import _NestedLaunchMixin


class FunctionTranslator(_EmitMixin, _NamingMixin, _PhasesMixin, _MagicLoopMixin, _HostRegionsMixin, _OuterVarsMixin, _StatementsMixin, _ReturnsMixin, _AssignMixin, _AugassignMixin, _TargetsMixin, _TuplesMixin, _PrintEmitMixin, _ExpressionsMixin, _OperatorsMixin, _SubscriptsMixin, _ImagesMixin, _AttributesMixin, _ComplexValuesMixin, _QuaternionsMixin, _MagicMethodsMixin, _VectorOpsMixin, _StructuresMixin, _CallsMixin, _BuiltinEmitMixin, _NumpyEmitMixin, _DeviceCallMixin, _RenderMixin, _BuiltinPythonMixin, _NdIndexMixin, _PrivateArrayMixin, _NestedLaunchMixin):

    # How many elements a private array created in the work-item body may spell out in its
    # initialiser; a longer one cannot be initialised without a loop, so it is refused.
    PRIVATE_ARRAY_MAX_UNROLL = 64


    # Element types a device atomic operation can address. Both backends provide 32-bit and
    # 64-bit integer atomics; int8/int16 have none, which is why a byte-sized histogram bin
    # is rejected rather than silently miscompiled.
    ATOMIC_ELEMENT_TYPES:Set[str] = {"int", "uint", "long", "ulong"}


    RETURN_CLAIM:str = "_ret_claim"

    RETURN_CLAIM:str = "_ret_claim"
    RETURN_SITE:str = "_ret_site"


    # The C element type a value slot is declared with, and the numpy dtype the host uses
    # for the same slot. A slot is internal storage, so its C type only has to hold the
    # value: bool travels as a byte (OpenCL rejects a _Bool kernel parameter and the CUDA
    # binding table has no bool entry), and char also travels as an unsigned byte because
    # OpenCL's `char` binds to ctypes.c_char (|S1), which is not an integer array at all.
    # The host converts when it reads the slot back.
    RETURN_SLOT_TYPES:Dict[str, Translated] = {
    "bool": ("uchar", np.uint8),
    "char": ("uchar", np.uint8),
    "uchar": ("uchar", np.uint8),
    "short": ("short", np.int16),
    "ushort": ("ushort", np.uint16),
    "int": ("int", np.int32),
    "uint": ("uint", np.uint32),
    "long": ("long", np.int64),
    "ulong": ("ulong", np.uint64),
    "float": ("float", np.float32),
    "double": ("double", np.float64),
    # A complex value's slot is one complex element for the same reason: the structure
    # the device declares is the complex layout, so a complex array *is* the buffer.
    "c64": ("c64", np.complex64),
    "c128": ("c128", np.complex128),
    }


    # The augmented operators that have a device atomic counterpart, and its gpu name.
    ATOMIC_ASSIGN_OPS:Dict[str, str] = {
    "Add": "atomic_add", "Sub": "atomic_sub",
    "BitAnd": "atomic_and", "BitOr": "atomic_or", "BitXor": "atomic_xor",
    }


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
                     type_map:Optional[Dict[str, str]]=None):
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
                tree:ast.Module = _parse_kernel_source(self.source_text)
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



_is_creation_expression = _module_is_creation_expression


__all__ = [
    "__init__",
    "_array_name",
    "_array_pointer_root",
    "_augassign_is_race_free",
    "_augassign_symbol",
    "_bind_carried_scalars",
    "_bind_host_statement",
    "_bind_host_value",
    "_bind_host_values",
    "_bind_outer_variables",
    "_buffer_qualifier",
    "_builtin_call_name",
    "_c_string_text",
    "_carried_scalars",
    "_class_of",
    "_clip_bound",
    "_collect_array_pointers",
    "_collect_constant_lists",
    "_complex_dtype_of",
    "_complex_operand",
    "_complex_side",
    "_complex_store",
    "_component",
    "_condition_code",
    "_constant_shape",
    "_convert_complex",
    "_define",
    "_device_func_name",
    "_device_qualifier",
    "_device_type",
    "_emit",
    "_emit_array_slice",
    "_emit_atomic_update",
    "_emit_blank",
    "_emit_block",
    "_emit_builtin",
    "_emit_clip",
    "_emit_complex_binop",
    "_emit_complex_builtin",
    "_emit_complex_compare",
    "_emit_complex_constructor",
    "_emit_complex_literal",
    "_emit_complex_math",
    "_emit_complex_method",
    "_emit_constant_enumerate_head",
    "_emit_cuda_only_builtin",
    "_emit_device_call",
    "_emit_element_loop",
    "_emit_floored",
    "_emit_generic_builtin",
    "_emit_gpu_builtin",
    "_emit_image_read",
    "_emit_image_write",
    "_emit_len",
    "_emit_library_function",
    "_emit_magic_binop",
    "_emit_magic_compare",
    "_emit_magic_conversion",
    "_emit_math_call",
    "_emit_method_on_value",
    "_emit_nd_head",
    "_emit_nested_launch",
    "_emit_numpy_cast",
    "_emit_numpy_elementwise",
    "_emit_numpy_elementwise_nodes",
    "_emit_print",
    "_emit_print_leaves",
    "_emit_private_array",
    "_emit_property_read",
    "_emit_property_write",
    "_emit_quaternion_binop",
    "_emit_quaternion_call",
    "_emit_scalar_relation",
    "_emit_self_method_call",
    "_emit_shared_math_builtin",
    "_emit_structure_constructor",
    "_emit_tuple_call_unpack",
    "_emit_vec_absdiff",
    "_emit_vec_builtin",
    "_emit_vec_clamp",
    "_emit_vec_constructor",
    "_emit_vec_cross",
    "_emit_vec_dot",
    "_emit_vec_hadd",
    "_emit_vec_mad",
    "_emit_vec_mix",
    "_emit_vec_normalize",
    "_emit_vec_relation",
    "_emit_vec_rhadd",
    "_emit_vec_select",
    "_emit_vec_swizzle",
    "_emit_vector_inplace",
    "_emit_where",
    "_emit_work_item_head",
    "_enumerate_magic_target",
    "_error",
    "_error_numpy_array_form",
    "_eval_constant",
    "_expand_print_pieces",
    "_extent_source",
    "_flatten_index",
    "_flattened_launch_id",
    "_folded_int_value",
    "_get_int_vec_type",
    "_handle_local_declaration",
    "_host_created_assignment",
    "_host_value_declaration",
    "_image_indices",
    "_image_param_type",
    "_image_sampler",
    "_image_vec_type",
    "_index_vector_axis",
    "_int_argument",
    "_int_format",
    "_is_block_id_expression",
    "_is_const_struct",
    "_is_creation_function",
    "_is_device_creation",
    "_is_launch_wide_name",
    "_is_local_buffer",
    "_is_numpy_constant_cast",
    "_is_numpy_dtype_name",
    "_is_print_call",
    "_is_scalar",
    "_is_secondary_magic_loop",
    "_is_shape0_range",
    "_is_structure_class",
    "_is_structure_operand",
    "_kernel_scalar_type",
    "_launch_id_code",
    "_library_call_name",
    "_local_names",
    "_lookup",
    "_lookup_type_of_index",
    "_loop_domain_name",
    "_loop_written_names",
    "_magic_owner",
    "_magic_receiver_type",
    "_member_is_bool",
    "_named_outer_target",
    "_nd_bound_code",
    "_nd_flat_index",
    "_nd_magic_target",
    "_ndindex_extents",
    "_needs_prologue_parameters",
    "_next_temp",
    "_non_atomic_writes",
    "_outer_argument",
    "_outer_param_declarations",
    "_outer_var_entry",
    "_pointer_source_name",
    "_print_argument",
    "_print_fstring",
    "_print_is_builtin",
    "_print_real_conversion",
    "_print_real_selector",
    "_print_real_text",
    "_print_sep_end",
    "_print_vec_pieces",
    "_private_element_type",
    "_private_initialiser",
    "_prologue_parameter_arguments",
    "_prologue_parameter_declarations",
    "_promote_returned_parameters",
    "_property_descriptor",
    "_push_scope_block",
    "_python_complex_kind",
    "_python_index",
    "_real_literal",
    "_record_element_write",
    "_record_member_write",
    "_record_return_slots",
    "_record_subscript_indices",
    "_referenced_names",
    "_reject_hip_image",
    "_reject_metal_image",
    "_render_body",
    "_render_nested_launch",
    "_reorder_keywords",
    "_require_complex",
    "_require_outer_parameter",
    "_require_quaternion_header",
    "_return_is_shared",
    "_return_param_type",
    "_return_position",
    "_return_site",
    "_return_slot",
    "_round_toward_zero_helper",
    "_sampler_name",
    "_scalar_position",
    "_shape0_range_param",
    "_shape_holder_param",
    "_shape_var",
    "_shared_math_call",
    "_shared_slice_shape",
    "_site_slots",
    "_slot_for_type",
    "_store_is_guarded",
    "_struct_member_access",
    "_structure_from_construction",
    "_structure_member_value",
    "_structure_operator_message",
    "_surface_name",
    "_target_is_bool_member",
    "_translate_magic_loop",
    "_translate_nested_kernel",
    "_translate_phase_kernels",
    "_trim_source",
    "_tuple_call_element_count",
    "_tuple_call_element_types",
    "_tuple_call_return_info",
    "_tuple_element_type",
    "_unsupported_vec_reason",
    "_vec_componentwise",
    "_vec_componentwise_compare",
    "_vec_maker",
    "_vec_type_name",
    "_visit_array_target",
    "_visit_assign",
    "_visit_atomic_augassign",
    "_visit_attribute",
    "_visit_augassign",
    "_visit_binop",
    "_visit_call",
    "_visit_compare",
    "_visit_constant",
    "_visit_enum_constant",
    "_visit_expression",
    "_visit_for",
    "_visit_host_statement",
    "_visit_kernel_return",
    "_visit_loop_exit",
    "_visit_name",
    "_visit_region",
    "_visit_return",
    "_visit_shared_index",
    "_visit_shared_slice",
    "_visit_sliced_subscript",
    "_visit_statement",
    "_visit_structure_constructor_call",
    "_visit_subscript",
    "_visit_subscript_index",
    "_visit_target",
    "_visit_unaryop",
    "_visit_work_item_body",
    "_why_not_callable",
    "_zero_value",
    "magic_loop_phases",
    "nests_a_launch",
    "parameter_declarations",
    "printf_backend",
    "render_device",
    "render_kernel",
    "render_kernels",
    "render_return_params",
    "render_structure_typedefs",
    "return_site_layout",
    "return_slot_dtype",
    "signature_digest",
    "structure_dtypes",
    "translate",
]
