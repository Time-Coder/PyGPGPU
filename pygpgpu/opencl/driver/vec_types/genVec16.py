"""The sixteen-component width base: pygf's genVec, with the width declared.

Like genVec8, this exists because pygf's GLSL-shaped surface stops at four
components while OpenCL's does not. The numeric swizzles name the slots above
nine in hex, either case (``sA`` and ``sa`` are both slot 10), which shadows
rgba's ``a`` and ``b`` on these classes exactly as OpenCL's layout does.
"""
from pygf.genVec import genVec as _PygfGenVec


class genVec16(_PygfGenVec):

    _numeric_swizzles = True

    def __len__(self)->int:
        return 16
