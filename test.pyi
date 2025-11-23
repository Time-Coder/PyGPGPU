from typing import override, overload, Tuple, Any
import concurrent.futures
import asyncio
import numpy as np
from numpy.typing import NDArray

from pygpgpu.opencl import *


class Kernel_flip_y(KernelWrapper):

    @override
    def __call__(self, src_image: image2d_t, dest_image: image2d_t, sampler: sampler_t)->None: ...

    @override
    def submit(self, src_image: image2d_t, dest_image: image2d_t, sampler: sampler_t)->concurrent.futures.Future: ...
    
    @override
    def async_call(self, src_image: image2d_t, dest_image: image2d_t, sampler: sampler_t)->asyncio.Future: ...

    def __getitem__(self, work_sizes:Tuple[int, int])->Kernel_flip_y: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int], Tuple[int]])->Kernel_flip_y: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int], Tuple[int, int]])->Kernel_flip_y: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int, int], Tuple[int, int, int]])->Kernel_flip_y: ...

    @overload
    def __getitem__(self, device:str)->Kernel_flip_y: ...


class Kernel_test(KernelWrapper):

    @override
    def __call__(self, a: NDArray[int2], length: int)->None: ...

    @override
    def submit(self, a: NDArray[int2], length: int)->concurrent.futures.Future: ...
    
    @override
    def async_call(self, a: NDArray[int2], length: int)->asyncio.Future: ...

    def __getitem__(self, work_sizes:Tuple[int, int])->Kernel_test: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int], Tuple[int]])->Kernel_test: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int], Tuple[int, int]])->Kernel_test: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int, int], Tuple[int, int, int]])->Kernel_test: ...

    @overload
    def __getitem__(self, device:str)->Kernel_test: ...


flip_y: Kernel_flip_y
test: Kernel_test
