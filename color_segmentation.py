import numpy as np
from skimage import img_as_float
from skimage.segmentation import felzenszwalb


PROTOTYPES_Ki67_RGB = {
    'positive': [(67, 30, 21), (108, 95, 89), (105, 86, 71), (82, 35, 7),
                 (103, 62, 30), (61, 45, 55), (90, 59, 39), (100, 63, 47)],
    'negative': [(105, 102, 119), (101, 100, 114), (111, 113, 125), (134, 133, 139),
                 (119, 116, 133), (108, 108, 118), (128, 127, 141), (100, 100, 112)],
    'background': [(200, 198, 201), (193, 189, 190), (174, 172, 177), (164, 162, 165),
                   (173, 178, 181), (165, 164, 169)],
}

def color_segmentation(image):
    segmentation_idcs = felzenszwalb(image, scale=1, sigma=0.8, min_size=20, multichannel=True)
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


def pixelwise_closest_centroid(image, centroids_rgb):
    # Centroids must be a dictionary { 'name of group': [(corresponding, rgb, triplets), ...] }
    # TODO: do this in the CIELAB color space
    distance = np.full(image.shape[:2] + (len(centroids_rgb),), fill_value=np.inf, dtype='float')
    image = img_as_float(image)
    for idx_c, list_reference_color_rgb in enumerate(centroids_rgb.values()):
        for reference_color_rgb in list_reference_color_rgb:
            reference_color_rgb = img_as_float(np.asarray([c/255 for c in reference_color_rgb]).reshape((1, 1, 3)))
            distance[:, :, idx_c] = np.minimum(
                distance[:, :, idx_c],
                np.sum(np.square(image - reference_color_rgb), axis=2))
    return np.argmin(distance, axis=2), distance
