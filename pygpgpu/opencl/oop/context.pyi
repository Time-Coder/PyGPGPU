from typing import Tuple, List, Optional, Dict, Any, Union

import numpy as np

from ..runtime import cl_context_properties, cl_mem_flags, cl_command_queue_properties
from .clobject import CLObject
from .device import Device
from .platform import Platform
from .program import Program
from .buffer import Buffer
from .command_queue import CommandQueue


class Context(CLObject):

    def __init__(self, *devices): ...

    def compile(self,
        file_name:str,
        includes:Optional[List[str]] = None,
        defines:Optional[Dict[str, Any]] = None,
        single_precision_constant:bool=False,
        denorms_are_zero:bool=False,
        fp32_correctly_rounded_divide_sqrt:bool=False,
        opt_disable:bool=False,
        strict_aliasing:bool=False,
        uniform_work_group_size:bool=False,
        no_subgroup_ifp:bool=False,
        mad_enable:bool=False,
        no_signed_zeros:bool=False,
        unsafe_math_optimizations:bool=False,
        finite_math_only:bool=False,
        fast_relaxed_math:bool=False,
        w:bool=False,
        Werror:bool=False,
        cl_std:Optional[float]=None,
        kernel_arg_info:bool=False,
        g:bool=False,
        create_library:bool=False,
        enable_link_options:bool=False,
        x_spir:bool=False,
        spir_std:Optional[float]=None
    )->Program:...

    def create_buffer(self, data:Union[bytes, bytearray, np.ndarray, None]=None, size:int=0, flags:Optional[cl_mem_flags]=None)->Buffer: ...

    def create_command_queue(self, device:Device, properties:Optional[cl_command_queue_properties]=None)->CommandQueue:...

    @property
    def devices(self)->Tuple[Device]: ...
    
    @property
    def platform(self)->Platform: ...

    @property
    def reference_count(self)->int: ...

    @property
    def num_devices(self)->int: ...

    @property
    def n_devices(self)->int: ...

    @property
    def device_ids(self)->List[int]: ...

    @property
    def properties(self)->List[cl_context_properties]: ...

    @property
    def d3d10_prefer_shared_resources(self)->bool: ...

    @property
    def d3d11_prefer_shared_resources(self)->bool: ...