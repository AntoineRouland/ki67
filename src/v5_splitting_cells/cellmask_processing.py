import numpy as np
from skimage.morphology import binary_closing, disk, binary_opening, diamond, binary_dilation


# Postprocess mask
###

def fill_holes(mask, max_radius):
    return binary_closing(mask, selem=disk(radius=max_radius))


def remove_thin_structures(mask, min_width):
    return binary_opening(mask, selem=disk(radius=min_width//2))


# Split cells
###

def manhattan_distance_to_mask(mask):
    if not np.any(mask):
        return np.full(shape=mask.shape, fill_value=np.inf)
    d = np.full(shape=mask.shape, fill_value=np.inf, dtype='int32')
    d_sofar = 0
    d[mask] = d_sofar
    se = diamond(1)
    while np.any(np.isinf(d)):
        d_sofar += 1
        mask = binary_dilation(mask, selem=se)
        reached_pixels = mask & ~np.isinf(d)
        d[reached_pixels] = d_sofar
    return d


def local_maxima_location(grayscale_2D_image):
    """ Pixel-wise local maxima of a 4-connected grid. """
    mask = np.full(shape=grayscale_2D_image.shape, fill_value=True)
    mask[0, :] = mask[:, 0] = mask[-1, :] = mask[:, -1] = False
    mask[:-1, :] &= (grayscale_2D_image[:-1, :] > grayscale_2D_image[+1:, :])
    mask[+1:, :] &= (grayscale_2D_image[+1:, :] > grayscale_2D_image[:-1, :])
    mask[:, :-1] &= (grayscale_2D_image[:, :-1] > grayscale_2D_image[:, +1:])
    mask[:, +1:] &= (grayscale_2D_image[:, +1:] > grayscale_2D_image[:, :-1])
    return mask

