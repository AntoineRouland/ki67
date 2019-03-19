import hashlib

import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_float
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import find_boundaries


# Simple Image Processing tasks
###

def apply_on_normalized_luminance(operation, image_rgb):
    image_lab = rgb2lab(image_rgb)
    luminance = image_lab[:, :, 0]
    a = np.min(luminance)
    b = np.max(luminance - a)
    luminance = (luminance - a)/b
    luminance = operation(luminance)
    luminance = luminance*b + a
    image_lab[:, :, 0] = luminance
    return lab2rgb(image_lab)


def crop(image, bounding_box):
        r_left, r_right, c_left, c_right = bounding_box
        return image[r_left:r_right, c_left:c_right, ...]


# Visualization
###


def visualize_classification(classification_as_indexed_labels):
    from src.v1_color_patches.patch_classifier import PROTOTYPES_Ki67_RGB
    classification_colored = np.empty(shape=classification_as_indexed_labels.shape + (3, ), dtype='float')
    for idx_class, list_ref_colors in enumerate(PROTOTYPES_Ki67_RGB.values()):
        region = classification_as_indexed_labels == idx_class
        color = np.asarray([c/255.0 for c in list_ref_colors[0]], dtype='float').reshape((1, 1, 3))
        classification_colored[region, :] = color
    return classification_colored


def outline_regions(image, region_labels):
    boundaries = find_boundaries(region_labels, mode='outer', background=0)
    image = img_as_float(image.copy())
    color = np.array([0, 1, 0], dtype='float').reshape((1, 1, 3))
    image[boundaries, :] = color
    return image


def average_color(image, region_labels):
    result_lab = rgb2lab(image)
    for label in range(np.max(region_labels)+1):
        region = region_labels == label
        result_lab[region, :] = np.average(result_lab[region, :], axis=0)
    return lab2rgb(result_lab)


def colormap(labels_mask):
    return plt.cm.get_cmap('tab20')(np.remainder(labels_mask, 20))[:, :, :3]


def visualize_contrasted_grayscale(grayscale_image):
    grayscale_image = grayscale_image - np.min(grayscale_image)
    if not np.all(grayscale_image == 0):
        grayscale_image = grayscale_image / np.max(grayscale_image)
    if grayscale_image.ndim == 2:
        grayscale_image = grayscale_image[:, :, np.newaxis]
    return np.repeat(grayscale_image, repeats=3, axis=2)


# Misc
###


def hash_np(numpy_array):
    return hashlib.sha1(numpy_array.view(np.uint8)).hexdigest()
