from camera import Camera
import matplotlib.pyplot as plt


cam = Camera()
image = cam.render()

plt.imshow(image)
plt.show()