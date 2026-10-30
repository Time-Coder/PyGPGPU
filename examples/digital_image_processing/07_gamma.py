from skimage import data
from matplotlib import pyplot as plt
from image_processing import gamma

image = data.coffee()

dest_image = gamma(image, 0.2)

plt.imshow(dest_image)
plt.show()