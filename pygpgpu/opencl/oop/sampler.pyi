from typing import List

from ..runtime import (
    cl_addressing_mode,
    cl_filter_mode,
    cl_sampler_properties,
    sampler_t
)

from .clobject import CLObject
from .context import Context


class sampler(CLObject):

    def __init__(self, context:Context, sampler_t_:sampler_t)->None: ...

    @property
    def context(self)->Context: ...
    
    @property
    def normalized_coords(self)->bool: ...
    
    @property
    def addressing_mode(self)->cl_addressing_mode: ...
    
    @property
    def filter_mode(self)->cl_filter_mode: ...

    @property
    def reference_count(self)->int: ...

    @property
    def properties(self)->List[cl_sampler_properties]: ...