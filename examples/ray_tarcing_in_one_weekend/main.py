from camera import Camera
from scene import Scene
from sphere import Sphere
import matplotlib.pyplot as plt
from pygf import point3f


# The book's framing: a tripod 1.2 m up and 3 m back, aimed at the ball. With the
# camera left at the origin and the 50 mm lens, the 0.5 m ball at z=-1 covers more
# than the whole viewport, so every ray hits it and the picture is a flat gradient.
cam = Camera()
cam.position = point3f(0, 1.2, 3.0)
cam.look_at(point3f(0, 0.0, -1))
cam.update_params()
scene = Scene()
scene.add(Sphere(point3f(0, 0, -1), 0.5))
scene.add(Sphere(point3f(0, -100.5, -1), 100))

image = cam.render(scene)

plt.imshow(image)
plt.show()
