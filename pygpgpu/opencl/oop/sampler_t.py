from ..runtime import (
    cl_addressing_mode,
    cl_filter_mode
)


class sampler_t:

    def __init__(self, normalized_coords:bool=False, addressing_mode:cl_addressing_mode=cl_addressing_mode.CL_ADDRESS_CLAMP_TO_EDGE, filter_mode:cl_filter_mode=cl_filter_mode.CL_FILTER_LINEAR)->None:
        self._normalized_coords:bool = normalized_coords
        self._addressing_mode:cl_addressing_mode = addressing_mode
        self._filter_mode:cl_filter_mode = filter_mode
    
    @property
    def normalized_coords(self)->bool:
        return self._normalized_coords
    
    @normalized_coords.setter
    def normalized_coords(self, normalized:bool)->None:
        self._normalized_coords = normalized
    
    @property
    def addressing_mode(self)->cl_addressing_mode:
        return self._addressing_mode
    
    @addressing_mode.setter
    def addressing_mode(self, mode:cl_addressing_mode)->None:
        self._addressing_mode = mode
    
    @property
    def filter_mode(self)->cl_filter_mode:
        return self._filter_mode
    
    @filter_mode.setter
    def filter_mode(self, mode:cl_filter_mode)->None:
        self._filter_mode = mode

    def __repr__(self)->str:
        return f"sampler_t({self._normalized_coords}, {self._addressing_mode}, {self._filter_mode})"
    
    def __str__(self)->str:
        return f"sampler_t({self._normalized_coords}, {self._addressing_mode}, {self._filter_mode})"