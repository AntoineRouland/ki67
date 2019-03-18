import datetime
import logging

import numpy as np
import os
import skimage.io as io
from skimage import img_as_float
from skimage.color import rgb2lab
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.transform import resize
from sklearn.cluster import KMeans

from src.data_loader import patient_names, images, root_dir, FOLDER_EXPERIMENTS
from src.utils import apply_on_normalized_luminance, colormap, outline_regions, average_color

MAX_PATIENTS = 1
MAX_IMAGES_PER_PATIENT = 1
MAX_PATCHES_PER_IMAGE = 2
RESIZE_IMAGES = None    # (1000, 1000)  # None to deactivate

if __name__ == "__main__":
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_dir = root_dir(FOLDER_EXPERIMENTS(version=4), execution_id)
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

            for gaussian_sigma in range(2, 12, 3):
                results_p_dir = os.path.join(results_dir, p_name, str(idx_img))
                os.makedirs(results_p_dir, exist_ok=True)
                logging.info(f'Processing: {p_name}-{idx_img}')

                if RESIZE_IMAGES:
                    logging.info('Resizing image')
                    image = resize(img_as_float(image), RESIZE_IMAGES + (3, ))
                io.imsave(fname=os.path.join(results_p_dir, '01 01 Original.jpg'),
                          arr=image)

                logging.info('Gaussian filter')
                image = apply_on_normalized_luminance(
                    operation=lambda img: gaussian(img, sigma=gaussian_sigma),
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
                range_centroids = list(range(2, 8))
                inertia = np.full(shape=(len(range_centroids), ),
                                  fill_value=np.inf)
                for idx, num_centroids in enumerate(range_centroids):
                    logging.info(f'{num_centroids:02d} centroids')
                    image_flat = rgb2lab(image).reshape((-1, 3))
                    clustering = KMeans(n_clusters=num_centroids, random_state=0).fit(image_flat)
                    io.imsave(fname=os.path.join(results_p_dir, f'02 K-means - {num_centroids:02d} centroids - labels.jpg'),
                              arr=colormap(clustering.labels_.reshape(image.shape[0:2])))
                    io.imsave(fname=os.path.join(results_p_dir, f'02 K-means - {num_centroids:02d} centroids - regions.jpg'),
                              arr=outline_regions(image=image, region_labels=clustering.labels_.reshape(image.shape[0:2])))
                    io.imsave(fname=os.path.join(results_p_dir, f'02 K-means - {num_centroids:02d} centroids - average_color.jpg'),
                              arr=average_color(image=image, region_labels=clustering.labels_.reshape(image.shape[0:2])))
                    inertia[idx] = np.min(inertia[idx], clustering.inertia_)

                eplot = EasyPlot(x, x ** 2, 'b-o', label='y = x**2', showlegend=True,
                                 xlabel='x', ylabel='y', title='title', grid='on')
                io.imsave(fname=os.path.join(results_p_dir,
                                             f'02 K-means - {num_centroids:02d} centroids - average_color.jpg'),
                          arr=average_color(image=image, region_labels=clustering.labels_.reshape(image.shape[0:2])))
