import numpy as np
from skimage import img_as_uint
from skimage.color import rgb2lab, lab2rgb


def apply_on_normalized_luminance(operation, image_rgb):
    image_lab = rgb2lab(img_as_uint(image_rgb))
    luminance = image_lab[:, :, 0]
    a = np.min(luminance)
    b = np.max(luminance - a)
    luminance = (luminance - a)/b
    luminance = operation(luminance)
    luminance = luminance*b + a
    image_lab[:, :, 0] = luminance
    return lab2rgb(image_lab)
