from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from image_processing import down_sampling

src_image: np.ndarray = data.coffee()
ratio: int = 20

dest_image = down_sampling(src_image, ratio)

plt.imshow(dest_image)
plt.show()