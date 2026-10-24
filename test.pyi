from typing import override, overload, Tuple
import ctypes
import concurrent.futures
import asyncio
import numpy as np
from numpy.typing import NDArray

from pygpgpu.opencl import *


class Point(ctypes.Structure):

    def __init__(self, x: float, y: float)->None: ...

    x: float
    y: float


class Circle(ctypes.Structure):

    def __init__(self, center: Point, radius: float)->None: ...

    center: Point
    radius: float


class Test(ctypes.Structure):

    def __init__(self, a: int, b: int, c: int, d: NDArray[np.int32], e: NDArray[np.int32], f: int, g: int, h: NDArray[np.int32], i: NDArray[np.int32])->None: ...

    a: int
    b: int
    c: int
    d: NDArray[np.int32]
    e: NDArray[np.int32]
    f: int
    g: int
    h: NDArray[np.int32]
    i: NDArray[np.int32]


class Test(ctypes.Structure):

    def __init__(self, a: int, b: int, c: int, d: NDArray[np.int32], e: NDArray[np.int32], f: int, g: int, h: NDArray[np.int32], i: NDArray[np.int32])->None: ...

    a: int
    b: int
    c: int
    d: NDArray[np.int32]
    e: NDArray[np.int32]
    f: int
    g: int
    h: NDArray[np.int32]
    i: NDArray[np.int32]


class Kernel_check_points_in_circle(KernelWrapper):

    @override
    def __call__(self, points: NDArray[Point], results: NDArray[np.int32], circle: Circle, num_points: int)->None: ...

    @override
    def submit(self, points: NDArray[Point], results: NDArray[np.int32], circle: Circle, num_points: int)->concurrent.futures.Future: ...
    
    @override
    def async_call(self, points: NDArray[Point], results: NDArray[np.int32], circle: Circle, num_points: int)->asyncio.Future: ...

    def __getitem__(self, work_sizes:Tuple[int, int])->Kernel_check_points_in_circle: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int], Tuple[int]])->Kernel_check_points_in_circle: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int], Tuple[int, int]])->Kernel_check_points_in_circle: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int, int], Tuple[int, int, int]])->Kernel_check_points_in_circle: ...

    @overload
    def __getitem__(self, device:str)->Kernel_check_points_in_circle: ...


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


check_points_in_circle: Kernel_check_points_in_circle
flipY: Kernel_flipY
