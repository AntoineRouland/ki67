import datetime
import logging

import numpy as np
import os
import skimage.io as io
from sklearn.cluster import KMeans, SpectralClustering
from skimage import img_as_float
from skimage.color import rgb2lab
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.transform import resize

from src.data_loader import patient_names, images, root_dir, FOLDER_EXPERIMENTS
from src.utils import apply_on_normalized_luminance, colormap, outline_regions, average_color

MAX_PATIENTS = 1
MAX_IMAGES_PER_PATIENT = 1
MAX_PATCHES_PER_IMAGE = 2
RESIZE_IMAGES = None    # (1000, 1000)  # None to deactivate

if __name__ == "__main__":
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_dir = root_dir(FOLDER_EXPERIMENTS(version=2), execution_id)
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    p_names = patient_names()
    for idx_p, p_name in enumerate(p_names[0:MAX_PATIENTS]):

        for idx_img, (path_image, image) in enumerate(images(patient_name=p_name, max_images=MAX_IMAGES_PER_PATIENT)):
            results_p_dir = os.path.join(results_dir, p_name, str(idx_img))
            os.makedirs(results_p_dir, exist_ok=True)
            logging.info(f'Processing: {p_name}-{idx_img}')

            if RESIZE_IMAGES:
                logging.info('Resizing image')
                image = resize(img_as_float(image), RESIZE_IMAGES + (3, ))
            io.imsave(fname=os.path.join(results_p_dir, '01 01 Original.jpg'),
                      arr=image)

            image_lab = rgb2lab(image)
            luminance = image_lab[:, :, 0]
            a = np.min(luminance)
            b = np.max(luminance - a)
            luminance = (luminance - a) / b
            io.imsave(fname=os.path.join(results_p_dir, '01 01 Luminance.png'),
                      arr=luminance)
            channel_a = image_lab[:, :, 1]
            a = np.min(channel_a)
            b = np.max(channel_a - a)
            channel_a = (channel_a - a) / b
            io.imsave(fname=os.path.join(results_p_dir, '01 01 channel_a.png'),
                      arr=channel_a)
            channel_b = image_lab[:, :, 2]
            a = np.min(channel_b)
            b = np.max(channel_b - a)
            channel_b = (channel_b - a) / b
            io.imsave(fname=os.path.join(results_p_dir, '01 01 channel_b.png'),
                      arr=channel_b)

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

            logging.info('K-means clustering')
            for num_centroids in range(3, 6):
                logging.info(f'{num_centroids:02d} centroids')
                image_flat = rgb2lab(image).reshape((-1, 3))
                clustering = KMeans(n_clusters=num_centroids, random_state=0).fit(image_flat)
                io.imsave(fname=os.path.join(results_p_dir, f'02 K-means - {num_centroids:02d} centroids - labels.jpg'),
                          arr=colormap(clustering.labels_.reshape(image.shape[0:2])))
                io.imsave(fname=os.path.join(results_p_dir, f'02 K-means - {num_centroids:02d} centroids - regions.jpg'),
                          arr=outline_regions(image=image, region_labels=clustering.labels_.reshape(image.shape[0:2])))
                io.imsave(fname=os.path.join(results_p_dir, f'02 K-means - {num_centroids:02d} centroids - average_color.jpg'),
                          arr=average_color(image=image, region_labels=clustering.labels_.reshape(image.shape[0:2])))

            logging.info('K-means clustering with spatial coordinates as well')
            for num_centroids in range(3, 6):
                logging.info(f'{num_centroids:02d} centroids')
                X, Y = np.meshgrid(range(image.shape[0]), range(image.shape[1]), indexing='ij')
                X = X / np.max(X)   # Normalize both coordinates
                Y = Y / np.max(Y)
                image_flat = np.concatenate([rgb2lab(image), X[:, :, np.newaxis], Y[:, :, np.newaxis]], axis=2).reshape((-1, 5))
                clustering = KMeans(n_clusters=num_centroids, random_state=0).fit(image_flat)
                io.imsave(fname=os.path.join(results_p_dir, f'03 K-means with X, Y - {num_centroids:02d} centroids - labels.jpg'),
                          arr=colormap(clustering.labels_.reshape(image.shape[0:2])))
                io.imsave(fname=os.path.join(results_p_dir, f'03 K-means with X, Y - {num_centroids:02d} centroids - regions.jpg'),
                          arr=outline_regions(image=image, region_labels=clustering.labels_.reshape(image.shape[0:2])))
                io.imsave(fname=os.path.join(results_p_dir, f'03 K-means with X, Y - {num_centroids:02d} centroids - average_color.jpg'),
                          arr=average_color(image=image, region_labels=clustering.labels_.reshape(image.shape[0:2])))
