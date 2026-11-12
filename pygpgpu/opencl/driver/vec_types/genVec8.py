"""The eight-component width base: pygf's genVec, with the width declared.

pygf carries no 8-wide base (its GLSL surface stops at four), so this subclass
declares the width over pygf's genVec and turns the OpenCL numeric swizzles on.
"""
from pygf.genVec import genVec as _PygfGenVec


class genVec8(_PygfGenVec):

    _numeric_swizzles = True

    def __len__(self)->int:
        return 8
