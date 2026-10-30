from camera import Camera
from scene import Scene
from sphere import Sphere
import matplotlib.pyplot as plt
from pygf import point3f


cam = Camera()
scene = Scene()
scene.add(Sphere(point3f(0, 0, -1), 0.5))
scene.add(Sphere(point3f(0, -100.5, -1), 100))

image = cam.render(scene)

plt.imshow(image)
plt.show()