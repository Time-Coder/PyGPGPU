from __future__ import annotations
import concurrent.futures
import asyncio
from typing import Dict, List, TYPE_CHECKING, Union, Tuple

from .kernel_info import ArgInfo

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

    def __getitem__(self, work_sizes:Tuple[Union[int, Tuple[int], Device], ...])->Kernel:
        old_work_sizes = work_sizes
        if not isinstance(work_sizes, tuple):
            work_sizes = (work_sizes,)

        if len(work_sizes) == 0:
            raise TypeError("KernelWrapper.__getitem__ takes at least 1 argument, 0 were given")
        elif len(work_sizes) == 1:
            device = work_sizes[0]
        elif len(work_sizes) == 2:
            device = Platforms[0].devices[0]
        elif len(work_sizes) == 3:
            device = work_sizes[2]
        else:
            raise TypeError(f"KernelWrapper.__getitem__ takes at most 3 argument, {len(work_sizes)} were given")
        
        if device not in self._kernels:
            self._kernels[device] = self._program_wrapper.program(device)[self._name]
            self._kernels[device].args = self._args
        
        return self._kernels[device][old_work_sizes]

    def __call__(self, *args, **kwargs)->None:
        return self[Platforms[0].devices[0]](*args, **kwargs)

    def submit(self, *args, **kwargs)->concurrent.futures.Future:
        return self[Platforms[0].devices[0]].submit(*args, **kwargs)
    
    def async_call(self, *args, **kwargs)->asyncio.Future:
        return self[Platforms[0].devices[0]].async_call(*args, **kwargs)

    @property
    def program_wrapper(self)->ProgramWrapper:
        return self._program_wrapper

    @property
    def name(self)->str:
        return self._name
    
    @property
    def args(self)->Dict[str, ArgInfo]:
        return self._args
        