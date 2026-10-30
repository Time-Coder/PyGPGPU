from hittable import Hittable, HitRecord
from pygf import point3f, vector3f, dot
from ray import Ray
import math


class Sphere(Hittable):

    def __init__(self, center: point3f, radius: float):
        self.center = center
        self.radius = radius

    def hit(self, ray: Ray, ray_tmin: float, ray_tmax: float)->HitRecord:
        oc: vector3f = self.center - ray.origin
        a: float = dot(ray.direction, ray.direction)
        h: float = dot(ray.direction, oc)
        c: float = dot(oc, oc) - self.radius ** 2

        discriminant: float = h**2 - a*c
        if discriminant < 0:
            return HitRecord(is_hitted=False)

        sqrtd: float = math.sqrt(discriminant)

        root: float = (h - sqrtd)/a
        if root <= ray_tmin or ray_tmax <= root:
            root = (h + sqrtd) / a
            if root <= ray_tmin or ray_tmax <= root:
                return HitRecord(is_hitted=False)

        t = root
        pos = ray.at(t)
        outward_normal = (pos - self.center) / self.radius
        record = HitRecord(
            is_hitted=True,
            t=t,
            position=pos
        )
        record.set_face_normal(ray, outward_normal)

        return record
