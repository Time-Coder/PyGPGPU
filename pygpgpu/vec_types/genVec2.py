"""The two-component width base: pygf's, with the OpenCL numeric swizzles on.

The base classes come from pygf -- one vector implementation serves both
libraries -- and this subclass carries the one thing PyGPGPU adds: the
``s0``-style numeric component spelling OpenCL's narrow and wide types use
(``char2().s01``, ``ushort8().s7``). pygf's own GLSL-shaped surface keeps that
gate off.
"""
from pygf.genVec2 import genVec2 as _PygfGenVec2


class genVec2(_PygfGenVec2):

    _numeric_swizzles = True
