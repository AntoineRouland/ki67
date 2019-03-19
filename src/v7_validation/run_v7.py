
import datetime
import logging
import re
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import skimage.io as io
from numpy.linalg import LinAlgError
from skimage import img_as_float
from skimage.color import rgb2lab
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.transform import resize
from sklearn.cluster import KMeans

from src.data_loader import sample_names, images, root_dir, FOLDER_EXPERIMENTS, get_expert_Ki67
from src.utils import apply_on_normalized_luminance, colormap, outline_regions, average_color, \
    visualize_contrasted_grayscale
from src.v5_splitting_cells.cellmask_processing import fill_holes, remove_thin_structures, manhattan_distance_to_mask, \
    local_maxima_location

MAX_PATIENTS = None
MAX_IMAGES_PER_PATIENT = None

SCALE = None  # None to deactivate
GAUSSIAN_FILTER_SD = 2
CLUSTERING_NUM_CENTROIDS = 4
RADIUS_FILL_HOLES = 3
WIDTH_REMOVE_THIN_STRUCTURES = 12

if __name__ == "__main__":
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_dir = root_dir(FOLDER_EXPERIMENTS(version=7), execution_id)
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    s_names = sample_names()
    df = None
    for idx_s, s_name in enumerate(s_names[0:MAX_PATIENTS]):

        Ki67_gt = get_expert_Ki67(s_name)

        for path_image, original_image in images(patient_name=s_name, max_images=MAX_IMAGES_PER_PATIENT):
            m = re.search(f'/{s_name}-(?P<n_img>[0-9]+).(?P<ext>[a-zA-Z]+)$', path_image)
            img_ext = m.group('ext')
            img_number = int(m.group('n_img'))
            img_name = f'{s_name}-{img_number}'
            img_filename = f'{img_name}.{img_ext}'
            results_p_dir = os.path.join(results_dir, s_name, img_name)
            os.makedirs(results_p_dir, exist_ok=True)

            result = {
                'execution_id': execution_id,
                'sample_name': s_name,
                'img_number': img_number,
                'img_file': img_filename,
                'Ki67_gt': Ki67_gt,
            }

            logging.info(f'Processing: {s_name}-{img_number}')

            if SCALE:
                logging.info('Resizing image')
                sz = [ceil(d*SCALE) for d in original_image.shape[:2]] + [3]
                original_image = resize(img_as_float(original_image), sz, mode='reflect', anti_aliasing=True)
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
            result['clustering_inertia'] = clustering.inertia_

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
            result['raw_positive_area_ratio'] = np.sum(positive_mask)/positive_mask.size
            result['raw_negative_area_ratio'] = np.sum(negative_mask)/positive_mask.size
            result['Ki67_from_raw_area_ratios'] = \
                result['raw_positive_area_ratio'] / (result['raw_positive_area_ratio']+result['raw_negative_area_ratio'])

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
            result['corrected_positive_area_ratio'] = np.sum(positive_mask)/positive_mask.size
            result['corrected_negative_area_ratio'] = np.sum(negative_mask)/positive_mask.size
            result['Ki67_from_corrected_area_ratios'] = \
                result['corrected_positive_area_ratio'] / (result['corrected_positive_area_ratio']+result['corrected_negative_area_ratio'])

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
            result['positive_count'] = np.sum(positive_cell_location)
            result['negative_count'] = np.sum(negative_cell_location)
            result['Ki67_from_count'] = result['positive_count'] / (result['positive_count']+result['negative_count'])

            if df is None:
                df = pd.DataFrame(columns=result.keys())
            df = df.append(pd.DataFrame(result, index=[result['img_file']]))
            df.to_excel(os.path.join(results_dir, 'df_all_results.xlsx'))
            df.to_csv(os.path.join(results_dir, 'df_all_results.csv'))

            try:
                sns.catplot(x='Ki67_gt', y='Ki67_from_count', data=df)
                plt.savefig(os.path.join(results_dir, 'df_catplot.png'))
                plt.close()
                plt.clf()

                fig, axes = plt.subplots(nrows=1, ncols=3)
                sns.violinplot(x='Ki67_gt', y="Ki67_from_raw_area_ratios", data=df, ax=axes[0])
                sns.violinplot(x='Ki67_gt', y="Ki67_from_corrected_area_ratios", data=df, ax=axes[1])
                sns.violinplot(x='Ki67_gt', y="Ki67_from_count", data=df, ax=axes[2])
                fig.tight_layout()
                plt.savefig(os.path.join(results_dir, 'df_violin.png'))
                plt.close()
                plt.clf()

                sns.pairplot(data=df, hue='Ki67_gt', vars=["Ki67_from_raw_area_ratios", "Ki67_from_corrected_area_ratios", "Ki67_from_count"])
                plt.savefig(os.path.join(results_dir, 'df_pairplot.png'))
                plt.close()
                plt.clf()

            except (ValueError, LinAlgError):
                pass

