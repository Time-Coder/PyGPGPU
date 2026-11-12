"""The four-component width base: pygf's, with the OpenCL numeric swizzles on."""
from pygf.genVec4 import genVec4 as _PygfGenVec4


class genVec4(_PygfGenVec4):

    _numeric_swizzles = True
