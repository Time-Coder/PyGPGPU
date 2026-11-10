from skimage import data
from image_viewer import imshow
from image_processing import gamma

image = data.coffee()

dest_image = gamma(image, 0.2)

imshow(dest_image)