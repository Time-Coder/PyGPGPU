from pygf import point3f, float3


class Ray:

    def __init__(self, origin: point3f, direction: float3):
        self.origin = origin
        self.direction = direction

    def at(self, t: float)->point3f:
        return self.origin + t * self.direction
    