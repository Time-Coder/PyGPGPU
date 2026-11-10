# The translator's type notation, the return-layout records, the reserved names.
from __future__ import annotations
import types
from typing import (Dict, List, Optional, Set, Tuple, Union)
from typing_extensions import TypedDict



PTR = "*"  # type notation: ("*", base) means an array parameter, a plain string means a scalar

# A local_array_per_group() slice: that group's storage, plus the shape it was declared with.
SHARED_SLICE = "shared_slice"

# A local_array()/shared_array() buffer: this block's own storage, indexed directly. The
# binding is ("blockarr", ctype, c_name, total, extents), the same five slots a per-group
# binding has, and the difference is the first one: a per-group buffer is a mapping that has
# to be indexed by its group, a block-local one already *is* the caller's block.
BLOCK_ARRAY = "blockarr"

# The *indexed* form of a block-local buffer, the counterpart of SHARED_SLICE: the C name of
# the storage, its element type, and the axes the subscripts still have to cover. The two
# markers stay distinct because an error has to name the spelling the body wrote, but every
# site that handles "a work-group buffer being indexed" handles both, which is WORK_GROUP_SLICES.
BLOCK_SLICE = "block_slice"

WORK_GROUP_SLICES:Tuple[str, ...] = (SHARED_SLICE, BLOCK_SLICE)

# Every marker that means "work-group storage is being indexed": the two slice forms above
# plus the block-local *binding*, which the store path sees directly when the base name is
# the target (`bins[i] = v`). `_visit_shared_index` normalizes the binding to its slice, so a
# site that handles indexing can accept all three and let that method sort out the shape.
WORK_GROUP_VALUES:Tuple[str, ...] = WORK_GROUP_SLICES + (BLOCK_ARRAY,)

# The builtin spellings that build each kind of work-group storage. Both pairs are published
# and each pair means one thing, so the translator decides the declaration kind from the name
# it resolved and the errors can name the spelling the user actually wrote.
PER_GROUP_BUILTINS:Tuple[str, str] = ("local_array_per_group", "shared_array_per_block")
BLOCK_LOCAL_BUILTINS:Tuple[str, str] = ("local_array", "shared_array")

# Names a kernel body may use that C, C++ or OpenCL C reserves. A parameter or local called
# `new` is legal python and illegal in every one of those languages -- CUDA and HIP compile
# as C++, so `void f(c64 new, ...)` fails with a bare `expected a ")"` that names nothing the
# user wrote. The mapping is the prefix `_pyg_` plus the name, which is the one spelling that
# cannot collide with a user's own identifier (a python name cannot contain the prefix
# without also being mapped, so the two never meet).
#
# The set is the union of C89/C99/C11 keywords, C++ keywords as far as CUDA C++ accepts
# them, and OpenCL C's address-space and function qualifiers (`__global`, `__kernel`, ...),
# because one generated body is compiled by all three.
RESERVED_DEVICE_NAMES:Set[str] = {
    # C and C++ keywords
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double",
    "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long",
    "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct",
    "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
    "bool", "catch", "class", "constexpr", "const_cast", "delete", "dynamic_cast",
    "explicit", "export", "false", "friend", "mutable", "namespace", "new", "noexcept",
    "nullptr", "operator", "private", "protected", "public", "reinterpret_cast",
    "static_cast", "template", "this", "throw", "true", "try", "typeid", "typename",
    "using", "virtual", "wchar_t",
    # C++20 spellings CUDA/HIP toolchains accept
    "concept", "constinit", "consteval", "co_await", "co_return", "co_yield",
    "requires", "char8_t", "char16_t", "char32_t",
    # OpenCL C qualifiers and types
    "__global", "__local", "__constant", "__private", "__kernel", "__read_only",
    "__write_only", "__read_write", "__generic", "__attribute__", "__kernel_exec",
}

RESERVED_PREFIX:str = "_pyg_"


def device_name(name:str)->str:
    """The C identifier a python name is emitted as.

    A name the device language reserves is spelled with the `_pyg_` prefix, so a kernel
    parameter called `new`, `register`, `float`, `global` or `kernel` compiles. Everything
    else is emitted unchanged, so the generated source still reads like the python it came
    from. The two spellings are disjoint by construction: `_pyg_new` is only produced for
    `new`, and a python identifier `_pyg_new` is itself prefixed (`_pyg__pyg_new`).
    """
    return f"{RESERVED_PREFIX}{name}" if name in RESERVED_DEVICE_NAMES else name


# The internal type names use OpenCL's spellings, because one generator serves every backend.
# CUDA C does not define uchar/ushort/uint/ulong (NVRTC rejects them; char/short/int/long
# and the full unsigned forms are fine), so the CUDA path writes those out in full. HIP
# compiles through CUDA generation too and its clang accepts the same full spellings.
_CUDA_TYPE_NAMES:Dict[str, str] = {
    "uchar": "unsigned char",
    "ushort": "unsigned short",
    "uint": "unsigned int",
    # `long` is 32-bit in CUDA C on Windows (where NVRTC and HIP are used here), so the
    # 64-bit spelling has to be written out: leaving it as `long` made an int64 array a
    # 4-byte-stride buffer, which silently produced wrong values for every element.
    "long": "long long",
    "ulong": "unsigned long long",
}

# An image parameter is ("image", channels, scalar type): its storage is opaque to
# the kernel, which only ever samples or stores through it.
IMAGE = "image"

# The type of a translated expression. A device type is one of a closed set of shapes:
#
# * a **scalar or vector name** -- `"int"`, `"float3"`, `"quatf"`, a struct name;
# * a **pointer** -- `("*", base)`, an array parameter;
# * an **image** -- `("image", channels, scalar)`, opaque storage the body samples;
# * a **shared slice** -- `("shared_slice", ...)`, one group's local_array_per_group buffer;
#
# `is_ptr` / `is_vec` / `is_image` / `is_int` decide which, so the union says exactly what
# those predicates accept. It used to be spelled `Any` at every one of these sites, which
# threw away the one piece of type information the translator carries around most.
DeviceType = Union[str, Tuple[object, ...]]

# A translated expression is its C text plus the type above -- the pair every `_visit_*`
# returns and every emitter consumes.
Translated = Tuple[str, DeviceType]

# One value slot a `return` fills: the C type the launch declares for it, and the device
# type the body produced. Two return sites whose values share a C type share the slot.
class ReturnLayoutEntry(TypedDict):
    ctype: str
    type: DeviceType

# One position of one `return` statement, as `_visit_kernel_return` sees it. A `value`
# position travels as a one-element buffer (`code` is how the body reads it back) and an
# `array` position travels as the array parameter itself, named by `name`. `ordinal` and
# `slot` appear once the slot map has assigned them.
class ReturnPosition(TypedDict, total=False):
    kind: str
    type: DeviceType
    code: str
    name: str
    ordinal: int
    slot: int
    slot_parameter: str

# One entry of `return_site_layout`: the positions a site hands back, in return order, and
# whether it returned a tuple (which the host unpacks rather than passes through).
class ReturnSiteLayout(TypedDict):
    positions: List[ReturnSitePosition]
    is_tuple: bool

# One host-facing rebuild entry: a `value` names the slot it reads (or, when the value
# already lives in a parameter, the parameter itself through `reused`), and an `array`
# names the parameter to hand back.
class ReturnSitePosition(TypedDict, total=False):
    kind: str
    ordinal: int
    slot: int
    name: str
    reused: str

# An outer-scope name the body reads: the device type it travels as, whether the body
# writes it (which makes it a one-element buffer), and how the host reaches it -- a
# closure cell, the kernel's module globals, or a helper's own module (`scope`).
class OuterVarEntry(TypedDict, total=False):
    type: DeviceType
    writable: bool
    kind: str
    scope: Optional[types.ModuleType]
    object: "OuterObjectEntry"

# An outer name whose python object becomes a synthesized structure: the `ctypes` class the
# host builds, the members the body reads (one per field, collected from every pass), and
# the member paths the body writes, which the host copies back after the launch.
class OuterObjectEntry(TypedDict):
    structure: type
    members: List[str]
    written: List[str]

# An object the body creates above the loop and the host therefore builds: the device
# type, plus the expression the host evaluates to make it.
class HostValueEntry(TypedDict):
    expr: str
    type: DeviceType
