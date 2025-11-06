from typing import Dict, List

from .device import Device
from .context import Context
from ..runtime import cl_build_status, cl_program_binary_type


class Program:

    def build(self)->None: ...

    @property
    def clean_cose(self)->str: ...

    @property
    def options(self)->List[str]: ...

    @property
    def reference_count(self)->int: ...

    @property
    def context(self)->Context: ...

    @property
    def num_devices(self)->int: ...

    @property
    def devices(self)->List[Device]: ...

    @property
    def source(self)->str: ...

    @property
    def il(self)->str: ...

    @property
    def il_khr(self)->str: ...

    @property
    def binary_sizes(self)->Dict[Device, int]: ...

    @property
    def binaries(self)->Dict[Device, bytes]: ...

    @property
    def num_kernels(self)->int: ...

    @property
    def kernel_names(self)->str: ...

    @property
    def scope_global_ctors_present(self)->bool: ...

    @property
    def scope_global_dtors_present(self)->bool: ...

    @property
    def build_status(self)->Dict[Device, cl_build_status]: ...

    @property
    def build_options(self)->Dict[Device, str]: ...
    
    @property
    def build_log(self)->Dict[Device, str]: ...
    
    @property
    def binary_type(self)->Dict[Device, cl_program_binary_type]: ...
    
    @property
    def global_variable_total_size(self)->Dict[Device, int]: ...
