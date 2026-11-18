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
        # The textbook recursion: the multiply lands after the callee answers, so the
        # bounce sequence is a real call stack on the backends whose device functions
        # may recurse (cuda/hip -- the launch raises the per-thread stack for it). The
        # scalar `0.5` multiplier is exact in binary floating point, so the recursive
        # order and the loop order agree bit for bit here; on the backends that forbid
        # recursion the same body keeps the refusal and the loop spelling below is the
        # portable form.
        record = self.hit(ray, Interval(0, math.inf))
        if record.is_hitted:
            if depth <= 0:
                return color3f(0)
            direction: vector3f = rand_vector3f_hemisphere(record.normal, tid, seed)
            return 0.5 * self.ray_color(Ray(record.position, direction), depth-1,
                                        tid, seed+1)

        unit_dir = normalize(ray.direction)
        weight = 0.5 * (unit_dir.y + 1.0)
        return mix(color3f(1), color3f(0.5, 0.7, 1.0), weight)
    