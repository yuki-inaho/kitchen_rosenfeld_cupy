import sys
import cupy as cp
import numpy as np
from typing import List
from cupy._core.core import ndarray as ndarray_cp
from cupyx.scipy.ndimage import sobel as sobel_cp

from timeit import timeit


def compute_derivatives_cp(image: ndarray_cp) -> List[ndarray_cp]:
    derivatives = [sobel_cp(image, axis=i) for i in range(image.ndim)]
    return derivatives


@timeit
def kitchen_rosenfeld_cupy(gray_cp: ndarray_cp) -> ndarray_cp:
    if gray_cp.ndim != 2:
        sys.exit("kitchen_rosenfeld_cp: gray scaled image is only supported")

    sobel_y_cp, sobel_x_cp = compute_derivatives_cp(gray_cp)
    sobel_yy_cp, sobel_yx_cp = compute_derivatives_cp(sobel_y_cp)
    sobel_xy_cp, sobel_xx_cp = compute_derivatives_cp(sobel_x_cp)

    numerator = sobel_xx_cp * sobel_y_cp ** 2 - 2 * sobel_xy_cp * sobel_x_cp * sobel_y_cp + sobel_yy_cp * sobel_x_cp ** 2
    denominator = sobel_x_cp ** 2 + sobel_y_cp ** 2
    kitchen_rosenfeld_cp = cp.zeros_like(gray_cp, dtype=float)
    mask = denominator != 0
    kitchen_rosenfeld_cp[mask] = numerator[mask] / denominator[mask]

    return kitchen_rosenfeld_cp
