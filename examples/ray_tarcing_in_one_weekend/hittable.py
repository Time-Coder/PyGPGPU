from pygf import point3f, vector3f, dot
from ray import Ray


class HitRecord:

    def __init__(self, is_hitted: bool = False, position: point3f = None,
                 normal: vector3f = None, t: float = 0.0,
                 front_face: bool = False):
        self.is_hitted = is_hitted
        self.position = position if position is not None else point3f()
        self.normal = normal if normal is not None else vector3f()
        self.t = t
        self.front_face = front_face

    def set_face_normal(self, ray: Ray, outward_normal: vector3f)->None:
        self.front_face: bool = dot(ray.direction, outward_normal)
        self.normal = (outward_normal if self.front_face else -outward_normal)


class Hittable:

    def hit(self, ray: Ray, ray_tmin: float, ray_tmax: float)->"HitRecord":
        raise NotImplementedError
