from pygf import point3f, float3
from ctypes import Structure


class Ray(Structure):

    _fields_ = [
        ("origin", point3f),
        ("direction", float3)
    ]

    def at(self, t: float)->point3f:
        return self.origin + t * self.direction
    