""""""
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
from ._exports import EXPORTED_NAMES
from .walk import _is_creation_expression
from .init import _InitMixin


class FunctionTranslator(_InitMixin, _EmitMixin, _NamingMixin, _PhasesMixin, _MagicLoopMixin, _HostRegionsMixin, _OuterVarsMixin, _StatementsMixin, _ReturnsMixin, _AssignMixin, _AugassignMixin, _TargetsMixin, _TuplesMixin, _PrintEmitMixin, _ExpressionsMixin, _OperatorsMixin, _SubscriptsMixin, _ImagesMixin, _AttributesMixin, _ComplexValuesMixin, _QuaternionsMixin, _MagicMethodsMixin, _VectorOpsMixin, _StructuresMixin, _CallsMixin, _BuiltinEmitMixin, _NumpyEmitMixin, _DeviceCallMixin, _RenderMixin, _BuiltinPythonMixin, _NdIndexMixin, _PrivateArrayMixin, _NestedLaunchMixin):
        PRIVATE_ARRAY_MAX_UNROLL = 64



        ATOMIC_ELEMENT_TYPES:Set[str] = {"int", "uint", "long", "ulong"}



        RETURN_CLAIM:str = "_ret_claim"



        RETURN_CLAIM:str = "_ret_claim"



        RETURN_SITE:str = "_ret_site"



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



        ATOMIC_ASSIGN_OPS:Dict[str, str] = {

        "Add": "atomic_add", "Sub": "atomic_sub",

        "BitAnd": "atomic_and", "BitOr": "atomic_or", "BitXor": "atomic_xor",

        }


__all__ = EXPORTED_NAMES
