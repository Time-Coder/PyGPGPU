from skimage import data
import numpy as np
from image_viewer import imshow
from image_processing import down_sampling

src_image: np.ndarray = data.coffee()
ratio: int = 20

dest_image = down_sampling(src_image, ratio)
imshow(dest_image, window_size=(src_image.shape[1], src_image.shape[0]))