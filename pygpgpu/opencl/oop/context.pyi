from typing import Tuple, List, Optional, Dict, Any, Union

import numpy as np

from ..runtime import cl_context_properties, cl_mem_flags, cl_command_queue_properties
from .clobject import CLObject
from .device import Device
from .platform import Platform
from .program import Program
from .buffer import Buffer
from .command_queue import CommandQueue
from .sampler import sampler
from .sampler_t import sampler_t
from .image2d import image2d
from .image2d_t import image2d_t
from .image1d import image1d
from .image1d_t import image1d_t
from .image3d import image3d
from .image3d_t import image3d_t
from .image2d_array import image2d_array
from .image2d_array_t import image2d_array_t
from .image1d_array import image1d_array
from .image1d_array_t import image1d_array_t


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

    def create_sampler(self, sampler_t_:sampler_t)->sampler: ...
    
    def create_image2d(self, image2d_t_:image2d_t)->image2d: ...

    def create_image1d(self, image:image1d_t)->image1d: ...
    
    def create_image3d(self, image:image3d_t)->image3d: ...
    
    def create_image2d_array(self, image:image2d_array_t)->image2d_array: ...
    
    def create_image1d_array(self, image:image1d_array_t)->image1d_array: ...

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