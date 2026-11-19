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
        # The recursive spelling. Each bounce is one self-call: the wrapper reads
        # only the slot and the names that keep one value for the walk, so the
        # translator takes it on every backend (opencl/metal included) -- the
        # level counter loop stands in for the recursion there, and the bounce
        # ray (a structure parameter the walk replaces per level) is carried by
        # the conversion's own local-value snapshot.
        # The walk's minimum hit distance is 0.01, not 0: a bounce ray leaves the
        # surface from a point ON it, and an interval starting at 0 accepts the
        # surface itself as an immediate hit (the book's shadow acne) -- the walk
        # then ping-pongs on its own surface and every pixel comes home dark. The
        # book's 0.001 is sized for its cosine-weighted sampler; the uniform
        # sampler this example uses produces grazing bounces that re-hit within
        # a millimetre, so the floor sits at a centimetre.
        record = self.hit(ray, Interval(0.01, math.inf))
        if record.is_hitted:
            if depth <= 0:
                return color3f(0)
            direction: vector3f = rand_vector3f_hemisphere(record.normal, tid, seed)
            return 0.5 * self.ray_color(
                Ray(record.position, direction), depth - 1, tid, seed + 1)
        unit_dir = normalize(ray.direction)
        weight = 0.5 * (unit_dir.y + 1.0)
        return mix(color3f(1), color3f(0.5, 0.7, 1.0), weight)
