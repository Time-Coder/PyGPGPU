from skimage import data
from image_viewer import imshow
from image_processing import adjust_brightness

image = data.coffee()
alpha = 5

dest_image = adjust_brightness(image, alpha)

imshow(dest_image)