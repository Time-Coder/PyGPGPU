from skimage import data
from image_viewer import imshow
from image_processing import scale

image = data.coffee()
alpha = 10

dest_image = scale(image, alpha)

imshow(dest_image)