from typing import override, overload, Tuple
import ctypes
import concurrent.futures
import asyncio
import numpy as np
from numpy.typing import NDArray

from pygpgpu.opencl import *


class Point3D(ctypes.Structure):

    def __init__(self, x: float, y: float, z: float)->None: ...

    x: float
    y: float
    z: float


class Particle(ctypes.Structure):

    def __init__(self, position: Point3D, velocity: Point3D, id: int, mass: float)->None: ...

    position: Point3D
    velocity: Point3D
    id: int
    mass: float


class Kernel_update_particle(KernelWrapper):

    @override
    def __call__(self, output: NDArray[float4], p: Particle)->None: ...

    @override
    def submit(self, output: NDArray[float4], p: Particle)->concurrent.futures.Future: ...
    
    @override
    def async_call(self, output: NDArray[float4], p: Particle)->asyncio.Future: ...

    def __getitem__(self, work_sizes:Tuple[int, int])->Kernel_update_particle: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int], Tuple[int]])->Kernel_update_particle: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int], Tuple[int, int]])->Kernel_update_particle: ...

    @overload
    def __getitem__(self, work_sizes:Tuple[Tuple[int, int, int], Tuple[int, int, int]])->Kernel_update_particle: ...

    @overload
    def __getitem__(self, device:str)->Kernel_update_particle: ...


update_particle: Kernel_update_particle
