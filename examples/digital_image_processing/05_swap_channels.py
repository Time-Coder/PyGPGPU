from skimage import data
from matplotlib import pyplot as plt
from image_processing import swap_channels

image = data.coffee()

dest_image = swap_channels(image, 2, 1, 0)

plt.imshow(dest_image)
plt.show()