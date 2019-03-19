import datetime
import logging
from math import ceil

import numpy as np
import os
import skimage.io as io
from skimage import img_as_float
from skimage.color import rgb2lab
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.transform import resize
from sklearn.cluster import KMeans

from src.data_loader import sample_names, images, root_dir, FOLDER_EXPERIMENTS
from src.utils import apply_on_normalized_luminance, colormap, outline_regions, average_color, \
    visualize_contrasted_grayscale
from src.v5_splitting_cells.cellmask_processing import fill_holes, remove_thin_structures, manhattan_distance_to_mask, \
    local_maxima_location

MAX_PATIENTS = 1
MAX_IMAGES_PER_PATIENT = 1
MAX_PATCHES_PER_IMAGE = 2

SCALE = None  # None to deactivate
GAUSSIAN_FILTER_SD = 2
CLUSTERING_NUM_CENTROIDS = 4
RADIUS_FILL_HOLES = 5
WIDTH_REMOVE_THIN_STRUCTURES = 12

if __name__ == "__main__":
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_dir = root_dir(FOLDER_EXPERIMENTS(version=5), execution_id)
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    p_names = sample_names()
    for idx_p, p_name in enumerate(p_names[0:MAX_PATIENTS]):

        for idx_img, (path_image, original_image) in enumerate(images(patient_name=p_name, max_images=MAX_IMAGES_PER_PATIENT)):

            results_p_dir = os.path.join(results_dir, p_name, str(idx_img))
            os.makedirs(results_p_dir, exist_ok=True)
            logging.info(f'Processing: {p_name}-{idx_img}')

            if SCALE:
                logging.info('Resizing image')
                sz = [ceil(d*SCALE) for d in original_image.shape[:2]] + [3]
                original_image = resize(img_as_float(original_image), sz)
            io.imsave(fname=os.path.join(results_p_dir, '01 01 Original.jpg'),
                      arr=original_image)
            image = original_image

            logging.info('Gaussian filter')
            image = apply_on_normalized_luminance(
                operation=lambda img: gaussian(img, sigma=GAUSSIAN_FILTER_SD),
                image_rgb=image)
            io.imsave(fname=os.path.join(results_p_dir, f'01 02 - Gaussian filter.jpg'),
                      arr=image)

            logging.info('CLAHE')
            image = apply_on_normalized_luminance(
                lambda img: equalize_adapthist(img, clip_limit=0.02),
                image_rgb=image)
            io.imsave(fname=os.path.join(results_p_dir, f'01 03 - CLAHE.jpg'),
                      arr=image)

            logging.info('K-means clustering')
            image_flat = rgb2lab(image).reshape((-1, 3))
            clustering = KMeans(n_clusters=CLUSTERING_NUM_CENTROIDS, random_state=0).fit(image_flat)
            io.imsave(fname=os.path.join(results_p_dir, f'02 K-means - labels.jpg'),
                      arr=colormap(clustering.labels_.reshape(image.shape[0:2])))
            io.imsave(fname=os.path.join(results_p_dir, f'02 K-means - regions.jpg'),
                      arr=outline_regions(image=original_image, region_labels=clustering.labels_.reshape(image.shape[0:2])))
            io.imsave(fname=os.path.join(results_p_dir, f'02 K-means - average_color.jpg'),
                      arr=average_color(image=original_image, region_labels=clustering.labels_.reshape(image.shape[0:2])))

            logging.info('Class separation')
            # Positive mask: cluster with maximum on channel a
            idx_positive_cluster = np.argmax(clustering.cluster_centers_[:, 1])
            positive_mask = np.equal(clustering.labels_, idx_positive_cluster).reshape(image.shape[0:2])
            # Negative mask: cluster with minimum on channel b
            idx_negative_cluster = np.argmin(clustering.cluster_centers_[:, 2])
            negative_mask = np.equal(clustering.labels_, idx_negative_cluster).reshape(image.shape[0:2])
            # Visualize
            io.imsave(fname=os.path.join(results_p_dir, f'03 Positives.jpg'),
                      arr=outline_regions(image=original_image, region_labels=positive_mask))
            io.imsave(fname=os.path.join(results_p_dir, f'03 Negatives.jpg'),
                      arr=outline_regions(image=original_image, region_labels=negative_mask))

            logging.info('Mask postprocessing')
            positive_mask = fill_holes(positive_mask, max_radius=RADIUS_FILL_HOLES)
            positive_mask = remove_thin_structures(positive_mask, min_width=WIDTH_REMOVE_THIN_STRUCTURES)
            negative_mask = fill_holes(negative_mask, max_radius=RADIUS_FILL_HOLES)
            negative_mask = remove_thin_structures(negative_mask, min_width=WIDTH_REMOVE_THIN_STRUCTURES)
            # Visualize
            io.imsave(fname=os.path.join(results_p_dir, f'04 Positives postprocessed.jpg'),
                      arr=outline_regions(image=original_image, region_labels=positive_mask))
            io.imsave(fname=os.path.join(results_p_dir, f'04 Negatives postprocessed.jpg'),
                      arr=outline_regions(image=original_image, region_labels=negative_mask))

            logging.info('Split cells')
            positive_pixelwise_radius = manhattan_distance_to_mask(mask=~positive_mask)
            positive_cell_location = local_maxima_location(positive_pixelwise_radius)
            negative_pixelwise_radius = manhattan_distance_to_mask(mask=~negative_mask)
            negative_cell_location = local_maxima_location(negative_pixelwise_radius)
            # Visualize
            io.imsave(fname=os.path.join(results_p_dir, f'05 Positives - Distance to border.jpg'),
                      arr=visualize_contrasted_grayscale(positive_pixelwise_radius))
            io.imsave(fname=os.path.join(results_p_dir, f'05 Positives - Locations border.jpg'),
                      arr=outline_regions(image=original_image, region_labels=positive_cell_location))
            io.imsave(fname=os.path.join(results_p_dir, f'05 Positives - Distance to border.jpg'),
                      arr=visualize_contrasted_grayscale(negative_pixelwise_radius))
            io.imsave(fname=os.path.join(results_p_dir, f'05 Positives - Locations border.jpg'),
                      arr=outline_regions(image=original_image, region_labels=negative_cell_location))

