from pygf import point3f, quatf, float3, color3f, conjugate
import numpy as np
from ray import Ray
from pygpgpu import kernel


class Camera:

    def __init__(self):
        self.position: point3f = point3f(0, 0, 0)
        self.orientation: quatf = quatf(1, 0, 0, 0)
        self.focal_length: float = 0.1
        self.aspect_ratio: float = 9/4
        self.image_width: int = 400
        self.viewport_width: float = 0.2
        self.update_params()

    def update_params(self):
        self.image_height: int = int(self.image_width / self.aspect_ratio)
        self.aspect_ratio: float = self.image_width / self.image_height
        self.viewport_height: float = self.viewport_width / self.aspect_ratio
        self.dx_in_view: float3 = float3(self.viewport_width / self.image_width, 0, 0)
        self.dy_in_view: float3 = float3(0, -self.viewport_height / self.image_height, 0)
        self.dx_in_world: float3 = self.view_dir_to_world(self.dx_in_view)
        self.dy_in_world: float3 = self.view_dir_to_world(self.dy_in_view)
        self.left_top_pixel_pos_in_view: point3f = (
            self.position + float3(
                -self.viewport_width/2 + 0.5*self.dx_in_view.x,
                self.viewport_height/2 - 0.5*self.dy_in_view.y,
                -self.focal_length
            )
        )
        self.left_top_pixel_pos_in_world: point3f = self.view_pos_to_world(self.left_top_pixel_pos_in_view)
        self.image: np.ndarray = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
    def world_pos_to_view(self, pos: point3f) -> point3f:
        return conjugate(self.orientation) * (pos - self.position)

    def world_dir_to_view(self, dir: float3) -> float3:
        return conjugate(self.orientation) * dir

    def view_pos_to_world(self, pos: point3f) -> point3f:
        return self.orientation * pos + self.position

    def view_dir_to_world(self, dir: float3) -> float3:
        return self.orientation * dir

    @kernel
    def render(self)->np.ndarray:
        for i, j in np.ndindex(self.image_height, self.image_width):
            hit_pixel_pos_in_world = (
                self.left_top_pixel_pos_in_world +
                (i * self.dy_in_world) +
                (j * self.dx_in_world)
            )
            ray_dir_in_world = hit_pixel_pos_in_world - self.position
            ray = Ray(self.position, ray_dir_in_world)
            pixel_color: color3f = self.ray_color(ray)
            self.write_color(i, j, pixel_color)

        return self.image

    def write_color(self, i:int, j:int, color:color3f):
        self.image[i, j, 0] = 255 * color.r
        self.image[i, j, 1] = 255 * color.g
        self.image[i, j, 2] = 255 * color.b

    def ray_color(self, ray: Ray)->color3f:
        return color3f()