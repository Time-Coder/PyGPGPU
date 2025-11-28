from typing import override, overload, Tuple
import ctypes
import concurrent.futures
import asyncio
import numpy as np
from numpy.typing import NDArray

from pygpgpu.opencl import *


class Kernel_flipY(KernelWrapper):

    @override
    def __call__(self, src_image: image2d_t, dest_image: image2d_t, s: sampler_t)->None: ...

    @override
    def submit(self, src_image: image2d_t, dest_image: image2d_t, s: sampler_t)->concurrent.futures.Future: ...
    
    @override
    def async_call(self, src_image: image2d_t, dest_image: image2d_t, s: sampler_t)->asyncio.Future: ...

    def __getitem__(self, work_sizes:Tuple[int, int])->Kernel_flipY: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int], Tuple[int]])->Kernel_flipY: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int], Tuple[int, int]])->Kernel_flipY: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int, int], Tuple[int, int, int]])->Kernel_flipY: ...

    @overload
    def __getitem__(self, device:str)->Kernel_flipY: ...


flipY: Kernel_flipY
