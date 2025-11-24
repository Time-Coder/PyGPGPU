from typing import override, overload, Tuple
import ctypes
import concurrent.futures
import asyncio
import numpy as np
from numpy.typing import NDArray

from pygpgpu.opencl import *


class Point(ctypes.Structure):

    _fields_ = [
        ('x', ctypes.c_float)
        ('y', ctypes.c_float)
        ('z', ctypes.c_float)
    ]

    def __init__(self, x: float, y: float, z: float)->None: ...

    x: float
    y: float
    z: float


class Kernel_compute_distance(KernelWrapper):

    @override
    def __call__(self, output: NDArray[np.float32], length: int, p: Point)->None: ...

    @override
    def submit(self, output: NDArray[np.float32], length: int, p: Point)->concurrent.futures.Future: ...
    
    @override
    def async_call(self, output: NDArray[np.float32], length: int, p: Point)->asyncio.Future: ...

    def __getitem__(self, work_sizes:Tuple[int, int])->Kernel_compute_distance: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int], Tuple[int]])->Kernel_compute_distance: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int], Tuple[int, int]])->Kernel_compute_distance: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int, int], Tuple[int, int, int]])->Kernel_compute_distance: ...

    @overload
    def __getitem__(self, device:str)->Kernel_compute_distance: ...


compute_distance: Kernel_compute_distance
