from pygf import (
    point3f,
    quatf,
    vector3f,
    color3f,
    conjugate,
    clamp,
    normalize,
    cross,
    dot,
)
import math
import numpy as np
from ray import Ray
from scene import Scene
from pygpgpu import kernel
from utils import rand


class Camera:

    def __init__(self):
        self.position: point3f = point3f(0, 0, 0)
        self.orientation: quatf = quatf(1, 0, 0, 0)
        # The camera's photographic parameters, in real units: the focal length and
        # the sensor width are *millimetres* (36 mm is a full-frame sensor's long
        # edge), and the focus distance is in *scene units* -- metres here, the same
        # unit the scene is built in. The viewport the rays are traced through lives
        # at the focus distance and spans sensor_width * focus_distance / focal_length
        # in scene units, which is the pinhole model: a 50 mm lens on full frame
        # focused at 2 m sees 1.44 m across, and the field of view it implies is a
        # property of the *ratio*, so the scene's unit never changes the framing as
        # long as the sensor, the lens and the scene distances are stated together.
        self.focal_length_mm: float = 50.0
        self.sensor_width_mm: float = 36.0
        self.focus_distance: float = 3.0
        self.aspect_ratio: float = 16/9
        self.image_width: int = 400
        self.samples_per_pixel: int = 100
        self.max_depth: int = 10
        self.update_params()

    def look_at(self, target: point3f, up: vector3f = vector3f(0, 1, 0)):
        """Aim the camera at ``target``, from its own position.

        The orientation is built in two steps, each a verifiable axis-angle
        quaternion: first the rotation that carries the camera's viewing direction
        (0, 0, -1) onto (target - position), then a roll about that direction which
        brings the camera's up as close to ``up`` as a rotation allows. Call it
        before `update_params`, because the viewport's world-space step vectors are
        derived from the orientation.
        """
        forward: vector3f = normalize(target - self.position)
        base: vector3f = vector3f(0, 0, -1)
        cos_turn: float = max(-1.0, min(1.0, -(forward.z)))
        turn: float = math.acos(cos_turn)
        if turn > 1e-9:
            axis: vector3f = normalize(cross(base, forward))
            half: float = turn / 2
            self.orientation = quatf(math.cos(half),
                                     axis.x * math.sin(half),
                                     axis.y * math.sin(half),
                                     axis.z * math.sin(half))
        else:
            # The forward direction is (anti-)parallel to the default: a 180-degree
            # turn needs an axis from the roll vector instead (the cross product
            # degenerates), and 0 degrees keeps the identity.
            if forward.z > 0:
                self.orientation = quatf(0, 0, 1, 0)
            else:
                self.orientation = quatf(1, 0, 0, 0)
        # The roll: where did the camera's own up land? Rotate (0, 1, 0) by the
        # orientation and measure the signed angle to the requested up, measured
        # about the forward direction.
        landed: vector3f = self.orientation * vector3f(0, 1, 0)
        projected_up: vector3f = normalize(up - forward * (up * forward))
        cos_roll: float = max(-1.0, min(1.0, dot(landed, projected_up)))
        roll: float = math.acos(cos_roll)
        # The sign: positive when landed sits clockwise from projected_up (as seen
        # looking against the forward direction) -- the cross of the two, dotted
        # with forward, says which way to roll.
        roll_sign: float = 1.0 if dot(cross(landed, projected_up), forward) > 0 else -1.0
        if roll > 1e-9:
            half: float = roll_sign * roll / 2
            self.orientation = (quatf(math.cos(half),
                                      forward.x * math.sin(half),
                                      forward.y * math.sin(half),
                                      forward.z * math.sin(half))
                                * self.orientation)

    def update_params(self):
        self.image_height: int = int(self.image_width / self.aspect_ratio)
        self.aspect_ratio: float = self.image_width / self.image_height
        # The viewport in scene units, at the focus plane (the pinhole model above).
        self.viewport_width: float = (self.sensor_width_mm
                                      * self.focus_distance / self.focal_length_mm)
        self.viewport_height: float = self.viewport_width / self.aspect_ratio
        self.du_in_view: vector3f = vector3f(self.viewport_width / self.image_width, 0, 0)
        self.dv_in_view: vector3f = vector3f(0, -self.viewport_height / self.image_height, 0)
        self.du_in_world: vector3f = self.view_dir_to_world(self.du_in_view)
        self.dv_in_world: vector3f = self.view_dir_to_world(self.dv_in_view)
        self.left_top_pixel_pos_in_view: point3f = (
            point3f(
                -self.viewport_width/2 + 0.5*self.du_in_view.x,
                self.viewport_height/2 - 0.5*self.dv_in_view.y,
                -self.focus_distance
            )
        )
        # The view-space point is the offset from the camera; `view_pos_to_world`
        # rotates it and adds the camera's position. Adding `position` here too
        # would count it twice -- invisible with the camera at the origin (the
        # original spelling), wrong the moment the camera moves.
        self.left_top_pixel_pos_in_world: point3f = self.view_pos_to_world(self.left_top_pixel_pos_in_view)
        self.image: np.ndarray = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
    def world_pos_to_view(self, pos: point3f) -> point3f:
        return conjugate(self.orientation) * (pos - self.position)

    def world_dir_to_view(self, dir: vector3f) -> vector3f:
        return conjugate(self.orientation) * dir

    def view_pos_to_world(self, pos: point3f) -> point3f:
        return self.orientation * pos + self.position

    def view_dir_to_world(self, dir: vector3f) -> vector3f:
        return self.orientation * dir

    def get_ray(self, i: int, j: int, sample: int)->Ray:
        offset: vector3f = self.sample_square(i, j, sample)
        pixel_sample = (
              self.left_top_pixel_pos_in_world
            + ((j + offset.x) * self.du_in_world)
            + ((i + offset.y) * self.dv_in_world)
        )
        ray_origin = self.position
        ray_direction = pixel_sample - ray_origin

        return Ray(ray_origin, ray_direction)

    def sample_square(self, i, j, seed):
        return vector3f(rand(i*self.image_width+j, seed)-0.5, rand(i*self.image_width+j, seed+1)-0.5, 0)

    @kernel
    def render(self, scene: Scene)->np.ndarray:
        for i, j in np.ndindex(self.image_height, self.image_width):
            tid: int = i*self.image_width + j
            pixel_color = color3f(0, 0, 0)
            for sample in range(0, self.samples_per_pixel):
                ray: Ray = self.get_ray(i, j, sample)
                pixel_color += scene.ray_color(ray, self.max_depth, tid, sample)

            self.write_color(i, j, pixel_color/self.samples_per_pixel)

        return self.image

    def write_color(self, i:int, j:int, color:color3f):
        clamp_color = clamp(color, color3f(0), color3f(1))
        self.image[i, j, 0] = 255 * clamp_color.r
        self.image[i, j, 1] = 255 * clamp_color.g
        self.image[i, j, 2] = 255 * clamp_color.b
