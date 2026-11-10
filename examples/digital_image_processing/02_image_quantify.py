from skimage import data
from image_viewer import imshow
from image_processing import quantify

image = data.coffee()
ratio = 128

dest_image = quantify(image, ratio)

imshow(dest_image)