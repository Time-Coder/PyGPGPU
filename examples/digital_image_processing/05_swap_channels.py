from skimage import data
from image_viewer import imshow
from image_processing import swap_channels

image = data.coffee()

dest_image = swap_channels(image, 2, 1, 0)

imshow(dest_image)