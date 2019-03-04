import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io
from skimage import img_as_float
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.morphology import opening, selem
from skimage.transform import resize

from color_segmentation import color_segmentation, PROTOTYPES_Ki67_RGB, pixelwise_closest_centroid
from data_loader import patient_names, images
from utils import apply_on_normalized_luminance

MAX_IMAGES_PER_PATIENT = 3
RESIZE_IMAGES = (500, 500)  # None to deactivate otherwise
COLOR = lambda img: plt.cm.get_cmap('tab20')(np.remainder(img, 20))[:, :, :3]

if __name__ == "__main__":
    results_dir = os.path.join('Results', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    for idx_p, p_name in enumerate(patient_names()):

        for idx_img, (path_image, image) in enumerate(images(patient_name=p_name, max_images=MAX_IMAGES_PER_PATIENT)):
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

            logging.info('Creating color segmentation')
            segmentation_idcs, segmentation_averaged = color_segmentation(image)
            io.imsave(fname=os.path.join(results_p_dir, '02 Segmentation idcs.jpg'),
                      arr=COLOR(segmentation_idcs))
            io.imsave(fname=os.path.join(results_p_dir, '03 Segmentation averaged.jpg'),
                      arr=segmentation_averaged)

            logging.info('Classifying based on fixed centroids')
            classification, distance_maps = pixelwise_closest_centroid(
                image=segmentation_averaged,
                centroids_rgb=PROTOTYPES_Ki67_RGB)
            io.imsave(fname=os.path.join(results_p_dir, '04 Classification.png'),
                      arr=COLOR(classification))
            io.imsave(fname=os.path.join(results_p_dir, '04 01 Positive.png'),
                      arr=COLOR(classification == 0))
            io.imsave(fname=os.path.join(results_p_dir, '04 02 Negative.jpg'),
                      arr=COLOR(classification == 1))
            io.imsave(fname=os.path.join(results_p_dir, '04 03 Background.jpg'),
                      arr=COLOR(classification == 2))

            logging.info('Visualizing classification')
            classification_colored = np.empty(shape=image.shape, dtype='float')
            for idx_class, list_ref_colors in enumerate(PROTOTYPES_Ki67_RGB.values()):
                for idx_c in range(3):
                    classification_colored[:, :, idx_c] = np.full(shape=classification.shape,
                                                                  fill_value=list_ref_colors[0][idx_c]/255)
            io.imsave(fname=os.path.join(results_p_dir, '04 04 Classification colored.jpg'),
                      arr=COLOR(classification_colored))

