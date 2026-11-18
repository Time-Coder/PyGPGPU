"""Render with the loop-spelled ray_color from a book-like viewpoint.

The camera sits 2 m above and 3 m behind the small sphere, aimed at it: the
50 mm lens on a full-frame sensor frames the ball against the ground sphere the
way the book's figure does.
"""
import sys
sys.path.insert(0, "examples\\ray_tarcing_in_one_weekend")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from camera import Camera
from scene import Scene
from sphere import Sphere
from pygf import point3f

cam = Camera()
cam.position = point3f(0, 1.2, 3.0)
cam.look_at(point3f(0, 0.0, -1))
cam.focal_length_mm = 50.0
cam.focus_distance = 3.0
cam.update_params()
scene = Scene()
scene.add(Sphere(point3f(0, 0, -1), 0.5))
scene.add(Sphere(point3f(0, -100.5, -1), 100))
image = cam.render["cuda"](scene)
plt.imsave("examples\\ray_tarcing_in_one_weekend\\recursive_render.png", image)
print("RENDERED", image.shape, flush=True)
