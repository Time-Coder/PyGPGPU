from pygf import point3f, vector3f


class Ray:

    def __init__(self, origin: point3f, direction: vector3f):
        self.origin = origin
        self.direction = direction

    def at(self, t: float)->point3f:
        return self.origin + t * self.direction
