from skimage import data
from matplotlib import pyplot as plt
from image_processing import scale

image = data.coffee()
alpha = 10

dest_image = scale(image, alpha)

plt.imshow(dest_image)
plt.show()