from pygf import point3f, vector3f, dot
from ray import Ray
from abc import ABC, abstractmethod
from interval import Interval


class HitRecord:

    def __init__(self, is_hitted: bool = False, position: point3f = None,
                 normal: vector3f = None, t: float = 0.0,
                 front_face: bool = False):
        self.is_hitted: bool = is_hitted
        self.position: point3f = position if position is not None else point3f()
        self.normal: vector3f = normal if normal is not None else vector3f()
        self.t: float = t
        self.front_face: bool = front_face

    def set_face_normal(self, ray: Ray, outward_normal: vector3f)->None:
        self.front_face: bool = dot(ray.direction, outward_normal)
        self.normal = (outward_normal if self.front_face else -outward_normal)


class Hittable(ABC):

    @abstractmethod
    def hit(self, ray: Ray, ray_range: Interval) -> HitRecord:
        raise NotImplementedError
