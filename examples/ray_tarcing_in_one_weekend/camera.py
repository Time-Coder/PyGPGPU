from pygf import (
    point3f,
    quatf,
    vector3f,
    color3f,
    conjugate,
    clamp,
    normalize,
    cross,
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

        The basis is the standard look-at: right = cross(forward, up_world)
        (horizontal for a vertical ``up``), true_up = cross(right, forward), and
        the orientation is the rotation whose columns carry the camera's local x/
        y/z onto that basis -- extracted from the orthonormal matrix with the
        trace form. Call it before `update_params`, because the viewport's
        world-space step vectors are derived from the orientation.
        """
        forward: vector3f = normalize(target - self.position)
        right: vector3f = normalize(cross(forward, up))
        true_up: vector3f = cross(right, forward)
        # The rotation matrix whose COLUMNS are the camera's local axes in world
        # space (local x -> right, local y -> true_up, local z -> -forward, since
        # the camera looks along its -z): R * v_local = v_world, which is the
        # convention pygf's `quat * vector` applies.
        m00, m10, m20 = right.x, right.y, right.z
        m01, m11, m21 = true_up.x, true_up.y, true_up.z
        m02, m12, m22 = -forward.x, -forward.y, -forward.z
        trace: float = m00 + m11 + m22
        if trace > 0.0:
            s: float = math.sqrt(trace + 1.0) * 2.0
            w: float = 0.25 * s
            x: float = (m21 - m12) / s
            y: float = (m02 - m20) / s
            z: float = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        self.orientation = quatf(w, x, y, z)

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
