from skimage import data
from matplotlib import pyplot as plt
from image_processing import adjust_brightness

image = data.coffee()
alpha = 5

dest_image = adjust_brightness(image, alpha)

plt.imshow(dest_image)
plt.show()