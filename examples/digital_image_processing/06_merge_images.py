from skimage import data
from image_viewer import imshow
from image_processing import merge_gray_images

moon = data.moon()
camera = data.camera()

image_minus = merge_gray_images(moon, camera, 1, -1)
image_plus = merge_gray_images(moon, camera, 1, 1)

imshow(image_minus)
