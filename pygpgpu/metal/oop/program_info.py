"""Per-argument metadata for a compiled Metal kernel.

This mirrors the CUDA `program_info` shape -- a `VarInfo` per parameter plus the
copy policy -- because Metal, like CUDA, has no OpenCL-style address qualifiers:
a pointer is just a pointer into a buffer, and the read/write analysis is what
says whether it travels to the device and comes back.

Two things differ from CUDA and are load-bearing:

* The type spellings here are **Metal Shading Language's** (`uint`, `uchar`,
  `ushort`, `ulong`, `half`, `uint3`...), not CUDA C's. MSL follows the C++
  standard spellings rather than CUDA's fixed-width conveniences, so the
  normalisation table maps both onto the one internal spelling the driver tables
  are keyed by (see `_device_type` in the translator).
* There are no texture/surface objects. A kernel reads an image through a
  `texture2d` parameter in MSL, which does exist, but `pygpgpu/image.py`'s Metal
  binding is a later step; until it lands an image argument is refused by the
  translator, not silently mis-typed here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from ... import numpy as gnp
from ...constants import KernelFlags

from .buffer import Buffer
from .event import Event

if TYPE_CHECKING:
    from .context import CommandQueue


# The two complex widths the kernel package declares as fixed C structures (`c64` is a
# `float` pair, `c128` a `double` pair -- see `kernel/complex.py`). A complex value is not a
# scalar, but MSL binds it exactly as one is bound: the generated signature declares
# `c64 value`, a value in the thread address space, so the host hands over the record's bytes
# with `setBytes:`. `check_type` therefore has to accept a complex number here, where the
# integer fallback below would refuse it with "must be an integer".
COMPLEX_TYPE_NAMES:Tuple[str, ...] = ("c64", "c128")


class VarInfo:

    # The driver tables are keyed by the OpenCL-style short names, while MSL writes
    # the full C++ spelling. Normalising here keeps every downstream lookup on one
    # spelling -- the same reason the CUDA path normalises `unsigned int` to `uint`.
    _TYPE_ALIASES:Dict[str, str] = {
        "unsigned char": "uchar",
        "unsigned short": "ushort",
        "unsigned int": "uint",
        "unsigned long": "ulong",
        "unsigned long long": "ulong",
        "long long": "long",
    }

    def __init__(self, name:str, type_str:str, array_shape:Tuple[int, ...],
                 type_qualifiers:Optional[list]=None,
                 kernel_flags:Optional[KernelFlags]=None)->None:
        self.name = name
        stripped:str = type_str.strip()
        pointer:str = ""
        if stripped.endswith("*"):
            stripped = stripped[:-1].rstrip()
            pointer = "*"
        self.type_str = VarInfo._TYPE_ALIASES.get(stripped, stripped) + pointer
        self.array_shape = array_shape
        self.type_qualifiers = list(type_qualifiers or [])
        self.kernel_flags = (KernelFlags.Readed | KernelFlags.Writed) if kernel_flags is None else kernel_flags
        self.value: object = None
        self.mem_obj: Optional[Buffer] = None

    @property
    def is_ptr(self)->bool:
        return self.type_str.endswith("*")

    @property
    def base_type_str(self)->str:
        return self.type_str[:-1] if self.is_ptr else self.type_str

    @property
    def writeonly(self)->bool:
        return False

    @property
    def readonly(self)->bool:
        return ("const" in self.type_qualifiers)

    @property
    def need_upload(self)->bool:
        """Whether this argument's host data has to travel to the device at all.

        A buffer the kernel neither reads nor writes contributes nothing and
        receives nothing, so both copies are pointless -- `KernelFlags(0)` says
        exactly that. Every other argument is uploaded, including a written one:
        the read-back is whole-buffer, so a buffer that was never uploaded would
        come back with zeros in every element the kernel left alone.
        """
        if not self.is_ptr:
            return True
        return bool(self.kernel_flags & (KernelFlags.Readed | KernelFlags.Writed))

    @property
    def need_read_back(self)->bool:
        return (self.is_ptr and bool(self.kernel_flags & KernelFlags.Writed))

    def use_buffer(self, cmd_queue:"CommandQueue",
                   data:np.ndarray)->Tuple[Buffer, Optional[Event]]:
        if isinstance(data, gnp.ndarray):
            used_buffer, event = data._to_device(cmd_queue, self.name,
                                                 kernel_flags=self.kernel_flags)
        else:
            used_buffer:Buffer = cmd_queue.context.get_buffer(data.nbytes)
            used_buffer.kernel_flags = self.kernel_flags
            # An argument the kernel never touches skips the copy entirely (see
            # need_upload); bind_data still records the read-back destination.
            if self.need_upload:
                event = used_buffer.set_data(cmd_queue, data)
            else:
                used_buffer.bind_data(data)
                event = None
        return used_buffer, event


class ArgInfo(VarInfo):

    def __init__(self, name:str, type_str:str, array_shape:Tuple[int, ...],
                 type_qualifiers:Optional[list]=None,
                 kernel_flags:Optional[KernelFlags]=None)->None:
        VarInfo.__init__(self, name, type_str, array_shape, type_qualifiers, kernel_flags)
        # True for a parameter carrying a dispatch attribute (thread_position_in_grid
        # and friends): the threadgroup dispatch fills that slot in, so the launch
        # must not bind a buffer there. The parser sets it; it stays False for every
        # parameter a handler .metal file declares without an attribute.
        self.dispatch_supplied:bool = False
        self.member_infos:Dict[str, "VarInfo"] = {}

    def check_type(self, value:object)->None:
        """Validate one launch argument against its declared device type."""
        base = self.base_type_str
        if self.is_ptr:
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"argument '{self.name}' must be np.ndarray, got {type(value).__name__}")
            return
        if base in ("float", "double", "half"):
            if not isinstance(value, (int, float, np.floating)):
                raise TypeError(
                    f"argument '{self.name}' must be a number, got {type(value).__name__}")
            return
        if base in COMPLEX_TYPE_NAMES:
            # A python `complex` and a numpy complex scalar are both the value a complex
            # parameter holds; anything else is a wrong argument, not a value MSL can read.
            if not isinstance(value, (complex, np.complexfloating)):
                raise TypeError(
                    f"argument '{self.name}' must be a complex number, got"
                    f" {type(value).__name__}")
            return
        if base in ("bool", "uchar", "char"):
            if not isinstance(value, (bool, int, np.bool_, np.integer)):
                raise TypeError(
                    f"argument '{self.name}' must be an integer, got {type(value).__name__}")
            return
        if not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"argument '{self.name}' must be an integer, got {type(value).__name__}")


class KernelInfo:
    """One entry point's parameters, name-matched to the device signature.

    `args` holds the parameters the **host binds**, in declaration order. The
    dispatch-supplied ones (the `[[thread_position_in_grid]]` family) live in
    `dispatch_parameters` instead: the device signature declares them, but the
    threadgroup dispatch fills them in, so listing them among `args` would put a
    parameter on the host's call signature that the caller cannot supply -- which is
    what `kernel_import` and the stub generator read `args` for.
    """

    def __init__(self, name:str)->None:
        self.name = name
        self.args: Dict[str, ArgInfo] = {}
        # Names of the trailing attribute parameters, in declaration order. Present in
        # the device signature, absent from the arguments a launch binds.
        self.dispatch_parameters: List[str] = []

    def args_declare(self, with_annotation:bool=False)->str:
        if with_annotation:
            return ", ".join(arg.name for arg in self.args.values())
        return ", ".join(list(self.args.keys()))

    def signature(self, with_annotation:bool=False)->str:
        return self.name + "(" + self.args_declare(with_annotation) + ")"
