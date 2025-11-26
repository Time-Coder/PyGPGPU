from __future__ import annotations
import concurrent.futures
import asyncio
from typing import Dict, List, TYPE_CHECKING, Union, Tuple

from .program_info import ArgInfo

if TYPE_CHECKING:
    from .program_wrapper import ProgramWrapper
    from .device import Device

from .kernel import Kernel
from .platforms import Platforms


class KernelWrapper:

    def __init__(self, program_wrapper:ProgramWrapper, name:str, args:List[ArgInfo], type_checked:bool)->None:
        self._program_wrapper:ProgramWrapper = program_wrapper
        self._name:str = name
        self._args:Dict[str, ArgInfo] = args
        self.type_checked:bool = type_checked
        self._kernels:Dict[Device, Kernel] = {}
        self._global_work_size = None
        self._local_work_size = None

    def __getitem__(self, work_sizes:Tuple[Union[int, Tuple[int], Device], ...])->Union[Kernel, KernelWrapper]:
        if not isinstance(work_sizes, tuple):
            work_sizes = (work_sizes,)

        if len(work_sizes) == 0:
            raise TypeError("KernelWrapper.__getitem__ takes at least 1 argument, 0 were given")
        elif len(work_sizes) == 1:
            device_str:str = work_sizes[0]
            device:Device = Platforms.device(device_str)
            if device not in self._kernels:
                self._kernels[device] = self._program_wrapper.program(device)[self._name]
                self._kernels[device].args = self._args

            kernel:Kernel = self._kernels[device][device_str]
            global_work_size = self._global_work_size
            local_work_size = self._local_work_size
            self._global_work_size = None
            self._local_work_size = None
            if global_work_size and local_work_size:
                kernel:Kernel = kernel[global_work_size, local_work_size]

            return kernel
        elif len(work_sizes) == 2:
            self._global_work_size = work_sizes[0]
            self._local_work_size = work_sizes[1]
            return self
        else:
            raise TypeError(f"KernelWrapper.__getitem__ takes at most 2 argument, {len(work_sizes)} were given")

    def _get_kernel(self)->Kernel:
        global_work_size = self._global_work_size
        local_work_size = self._local_work_size
        self._global_work_size = None
        self._local_work_size = None
        kernel:Kernel = self[Platforms[0].devices[0].name]
        if global_work_size and local_work_size:
            kernel:Kernel = kernel[global_work_size, local_work_size]

        return kernel

    def __call__(self, *args, **kwargs)->None:
        kernel:Kernel = self._get_kernel()
        return kernel(*args, **kwargs)

    def submit(self, *args, **kwargs)->concurrent.futures.Future:
        kernel:Kernel = self._get_kernel()
        return kernel.submit(*args, **kwargs)
    
    def async_call(self, *args, **kwargs)->asyncio.Future:
        kernel:Kernel = self._get_kernel()
        return kernel.async_call(*args, **kwargs)

    @property
    def program_wrapper(self)->ProgramWrapper:
        return self._program_wrapper

    @property
    def name(self)->str:
        return self._name
    
    @property
    def args(self)->Dict[str, ArgInfo]:
        return self._args
        