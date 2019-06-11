import numpy as np
from skimage.segmentation import felzenszwalb


def color_segmentation(image):
    segmentation_idcs = felzenszwalb(image, scale=0.5, sigma=0.8, min_size=10, multichannel=True)
    unique_idcs = np.unique(segmentation_idcs)
    flattened_image = image.reshape((-1, 3))
    image_averaged = np.empty(flattened_image.shape)
    for idx in unique_idcs:
        selected_region_idcs = np.equal(segmentation_idcs, idx).flatten()
        selected_region = flattened_image[selected_region_idcs, :]
        # TODO: do this in the CIELAB color space
        average_color = np.mean(selected_region, axis=0)
        image_averaged[selected_region_idcs, :] = average_color[:]
    image_averaged = image_averaged.reshape(image.shape)
    return segmentation_idcs, image_averaged
