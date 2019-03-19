import datetime
import logging

import numpy as np
import os
import skimage.io as io
from sklearn.cluster import KMeans
from skimage import img_as_float
from skimage.color import rgb2lab
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.transform import resize

from src.data_loader import sample_names, images, root_dir, FOLDER_EXPERIMENTS
from src.utils import apply_on_normalized_luminance, colormap, outline_regions, average_color

MAX_PATIENTS = 1
MAX_IMAGES_PER_PATIENT = 1
MAX_PATCHES_PER_IMAGE = 2
RESIZE_IMAGES = None    # (300, 300)  # None to deactivate

if __name__ == "__main__":
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_dir = root_dir(FOLDER_EXPERIMENTS(version=3), execution_id)
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

        for idx_img, (path_image, image) in enumerate(images(patient_name=p_name, max_images=MAX_IMAGES_PER_PATIENT)):
            results_p_dir = os.path.join(results_dir, p_name, str(idx_img))
            os.makedirs(results_p_dir, exist_ok=True)
            logging.info(f'Processing: {p_name}-{idx_img}')

            # 01 Preprocessing
            ###

            if RESIZE_IMAGES:
                logging.info('Resizing image')
                image = resize(img_as_float(image), RESIZE_IMAGES + (3, ))
            io.imsave(fname=os.path.join(results_p_dir, '01 01 Original.jpg'),
                      arr=image)

            logging.info('Gaussian filter')
            image = apply_on_normalized_luminance(
                operation=gaussian,
                image_rgb=image)
            io.imsave(fname=os.path.join(results_p_dir, '01 02 Gaussian filter.jpg'),
                      arr=image)

            logging.info('CLAHE')
            image = apply_on_normalized_luminance(
                lambda img: equalize_adapthist(img, clip_limit=0.02),
                image_rgb=image)
            io.imsave(fname=os.path.join(results_p_dir, '01 03 CLAHE.jpg'),
                      arr=image)

            # 02 Lab visualization
            ###

            image_lab = rgb2lab(image)
            luminance = image_lab[:, :, 0]
            a = np.min(luminance)
            b = np.max(luminance - a)
            luminance = (luminance - a) / b
            io.imsave(fname=os.path.join(results_p_dir, '02 01 Luminance.png'),
                      arr=luminance)
            channel_a = image_lab[:, :, 1]
            a = np.min(channel_a)
            b = np.max(channel_a - a)
            channel_a = (channel_a - a) / b
            io.imsave(fname=os.path.join(results_p_dir, '02 01 channel_a.png'),
                      arr=channel_a)
            channel_b = image_lab[:, :, 2]
            a = np.min(channel_b)
            b = np.max(channel_b - a)
            channel_b = (channel_b - a) / b
            io.imsave(fname=os.path.join(results_p_dir, '02 01 channel_b.png'),
                      arr=channel_b)

            # 03-04 Lab separation
            ###

            logging.info('Positive separation (k-means clustering on `a channel`)')
            channel_a = image_lab[:, :, 1]
            clustering = KMeans(n_clusters=2, random_state=0).fit(channel_a.reshape(-1, 1))
            idx_positive_cluster = np.argmax(clustering.cluster_centers_)
            positive_mask = np.equal(clustering.labels_, idx_positive_cluster).reshape(channel_a.shape[0:2])
            io.imsave(fname=os.path.join(results_p_dir, f'03 K-means - Positives - Mask.jpg'),
                      arr=colormap(positive_mask))
            io.imsave(fname=os.path.join(results_p_dir, f'03 K-means - Positives - regions.jpg'),
                      arr=outline_regions(image=image, region_labels=positive_mask))
            io.imsave(fname=os.path.join(results_p_dir, f'03 K-means - Positives - average color.jpg'),
                      arr=average_color(image=image, region_labels=positive_mask))

            logging.info('Negative separation (k-means clustering on `b channel`)')
            channel_b = image_lab[:, :, 2]
            nonpositive_mask = np.logical_not(positive_mask)
            clustering = KMeans(n_clusters=2, random_state=0).fit(channel_b[nonpositive_mask].reshape(-1, 1))
            idx_negative_cluster = np.argmin(clustering.cluster_centers_)
            negative_mask_wrt_nonpositive = np.equal(clustering.labels_, idx_negative_cluster)
            negative_mask = np.full(shape=positive_mask.shape, dtype=bool, fill_value=False)
            nonpositive_mask_as_flatten_idcs = np.where(nonpositive_mask.flatten())
            negative_mask_wrt_nonpositive_flatten = nonpositive_mask_as_flatten_idcs[0][negative_mask_wrt_nonpositive]
            negative_mask = negative_mask.flatten()
            negative_mask[negative_mask_wrt_nonpositive_flatten] = True
            negative_mask = negative_mask.reshape(positive_mask.shape)
            io.imsave(fname=os.path.join(results_p_dir, f'04 K-means - Negatives - Mask.jpg'),
                      arr=colormap(negative_mask))
            io.imsave(fname=os.path.join(results_p_dir, f'04 K-means - Negatives - regions.jpg'),
                      arr=outline_regions(image=image, region_labels=negative_mask))
            io.imsave(fname=os.path.join(results_p_dir, f'04 K-means - Negatives - average color.jpg'),
                      arr=average_color(image=image, region_labels=negative_mask))

            # 05 Results visualization
            ###

            region_labels = negative_mask + 2*positive_mask
            io.imsave(fname=os.path.join(results_p_dir, '05 Results - labels.jpg'),
                      arr=colormap(region_labels))
            io.imsave(fname=os.path.join(results_p_dir, '05 Results - regions.jpg'),
                      arr=outline_regions(image=image, region_labels=region_labels))
            io.imsave(fname=os.path.join(results_p_dir, '05 Results - average_color.jpg'),
                      arr=average_color(image=image, region_labels=region_labels))
