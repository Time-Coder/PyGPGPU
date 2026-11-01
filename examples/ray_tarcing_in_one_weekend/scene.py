from hittable import Hittable, HitRecord
from ray import Ray
from pygf import color3f, normalize, mix
import math
from typing import override
from interval import Interval


class Scene(Hittable):

    def __init__(self):
        self.hittables: list[Hittable] = []

    def clear(self):
        self.hittables.clear()

    def add(self, object: Hittable):
        self.hittables.append(object)

    @override
    def hit(self, ray: Ray, ray_range: Interval) -> HitRecord:
        record = HitRecord(is_hitted=False)
        closest_so_far: float = ray_range.max_value

        for hitable in self.hittables:
            temp_record = hitable.hit(ray, Interval(ray_range.min_value, closest_so_far)) 
            if temp_record.is_hitted:
                closest_so_far = temp_record.t
                record = temp_record

        return record

    def ray_color(self, ray: Ray)->color3f:
        record = self.hit(ray, Interval(0, math.inf))
        if record.is_hitted:
            return 0.5 * (record.normal + color3f(1))

        unit_dir = normalize(ray.direction)
        weight = 0.5 * (unit_dir.y + 1.0)
        return mix(color3f(1), color3f(0.5, 0.7, 1.0), weight)
    