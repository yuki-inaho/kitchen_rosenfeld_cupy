from skimage import img_as_float
from skimage.data import astronaut
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

import cupy as cp
from operators import kitchen_rosenfeld_cupy


# Load a sample image
color_image = astronaut()
color_image_normalized = img_as_float(color_image)
gray = rgb2gray(color_image_normalized)

# Calculate kitchen-rosenfeld measure
gray_cp = cp.asarray(gray)
kitchen_rosenfeld_cp = kitchen_rosenfeld_cupy(gray_cp)
kitchen_rosenfeld = cp.asnumpy(kitchen_rosenfeld_cp)

# Drawing
fig, ax = plt.subplots(ncols=2, nrows=1)
ax[0].imshow(color_image)
ax[0].set_title('Raw')
ax[1].imshow(kitchen_rosenfeld)
ax[1].set_title('Kitchen-Rosenfeld')
fig.tight_layout()
plt.show()