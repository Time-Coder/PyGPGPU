from hittable import Hittable, HitRecord
from ray import Ray
from pygf import color3f, normalize, mix, vector3f
from utils import rand_vector3f_hemisphere
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

    def ray_color(self, ray: Ray, depth: int, tid: int, seed: int)->color3f:
        # The bounce recursion is not a tail call (the 0.5 * lands after the callee
        # answers), so the loop is spelled by hand: one iteration is one bounce, and
        # `current` is rebuilt as a value each round -- naming the pointer parameter
        # directly would make the local a pointer alias, which cannot be reassigned.
        current: Ray = Ray(ray.origin, ray.direction)
        attenuation: float = 1.0
        for bounce in range(depth):
            record = self.hit(current, Interval(0, math.inf))
            if record.is_hitted:
                attenuation *= 0.5
                direction: vector3f = rand_vector3f_hemisphere(record.normal, tid, seed)
                seed += 1
                current = Ray(record.position, direction)
            else:
                unit_dir = normalize(current.direction)
                weight = 0.5 * (unit_dir.y + 1.0)
                return attenuation * mix(color3f(1), color3f(0.5, 0.7, 1.0), weight)

        return color3f(0)
    