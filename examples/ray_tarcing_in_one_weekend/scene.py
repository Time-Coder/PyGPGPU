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
        # The bounce walk is spelled as a loop: one iteration is one bounce, and
        # `current` is rebuilt as a value each round -- naming the pointer parameter
        # directly would make the local a pointer alias, which cannot be reassigned.
        # The walk's minimum hit distance is 0.01, not 0: a bounce ray leaves the
        # surface from a point ON it, and an interval starting at 0 accepts the
        # surface itself as an immediate hit (the book's shadow acne) -- the walk
        # then ping-pongs on its own surface and every pixel comes home dark. The
        # book's 0.001 is sized for its cosine-weighted sampler; the uniform
        # sampler this example uses produces grazing bounces that re-hit within
        # a millimetre, so the floor sits at a centimetre.
        current: Ray = Ray(ray.origin, ray.direction)
        attenuation: float = 1.0
        for bounce in range(depth):
            record = self.hit(current, Interval(0.01, math.inf))
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
    