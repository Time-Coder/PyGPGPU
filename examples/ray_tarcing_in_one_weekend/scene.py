from hittable import Hittable, HitRecord
from ray import Ray
from pygf import color3f, normalize, mix
import math


class Scene:

    def __init__(self):
        self.hittables: list[Hittable] = []

    def clear(self):
        self.hittables.clear()

    def add(self, object: Hittable):
        self.hittables.append(object)

    def hit(self, ray: Ray, ray_tmin: float, ray_tmax: float)->HitRecord:
        record = HitRecord(is_hitted=False)
        closest_t: float = ray_tmax

        for hitable in self.hittables:
            temp_record = hitable.hit(ray, ray_tmin, closest_t) 
            if temp_record.is_hitted:
                closest_t = temp_record.t
                record = temp_record

        return record

    def ray_color(self, ray: Ray)->color3f:
        record = self.hit(ray, 0, math.inf)
        if record.is_hitted:
            return 0.5 * (record.normal + color3f(1))

        unit_dir = normalize(ray.direction)
        weight = 0.5 * (unit_dir.y + 1.0)
        return mix(color3f(1), color3f(0.5, 0.7, 1.0), weight)
    