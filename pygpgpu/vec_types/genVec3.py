"""The three-component width base: pygf's, with the OpenCL numeric swizzles on.

The width is pygf's fixed 3, not ``len(_fields_)``: the OpenCL driver class pads
a vec3 to four slots (a trailing ``_`` field), and every width-driven answer --
construction, iteration, swizzles, arithmetic -- must still see three
components.
"""
from pygf.genVec3 import genVec3 as _PygfGenVec3


class genVec3(_PygfGenVec3):

    _numeric_swizzles = True
