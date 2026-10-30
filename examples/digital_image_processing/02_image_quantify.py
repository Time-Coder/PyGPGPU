from skimage import data
from matplotlib import pyplot as plt
from image_processing import quantify

image = data.coffee()
ratio = 128

dest_image = quantify(image, ratio)

plt.imshow(dest_image)
plt.show()