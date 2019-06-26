import datetime
import logging

import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.color import rgb2lab

from src.data_loader import root_dir, FOLDER_EXPERIMENTS
from src.v8_test.segmentation import segmentation
from src.v8_test.validation import validation
from src.utils import apply_on_normalized_luminance


MAX_PATIENTS = 1
MAX_IMAGES_PER_PATIENT = 1
MAX_PATCHES_PER_IMAGE = 2


if __name__ == "__main__":

    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_dir = root_dir(FOLDER_EXPERIMENTS(version=8), execution_id)
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    image_original = io.imread("/home/uib/PycharmProjects/ki67/Data/1309620 -A2/1309620 -A2-27.jpg")

    results_p_dir = os.path.join(results_dir, "image")
    os.makedirs(results_p_dir, exist_ok=True)
    io.imsave(fname=os.path.join(results_p_dir, '01 00 Original.jpg'),
              arr=image_original)

    logging.info('Gaussian filter')
    image = apply_on_normalized_luminance(
        operation=lambda img: gaussian(img, sigma=2),
        image_rgb=image_original)
    io.imsave(fname=os.path.join(results_p_dir, f'01 01 - Gaussian filter.jpg'),
              arr=image)

    logging.info('CLAHE')
    image = apply_on_normalized_luminance(
        lambda img: equalize_adapthist(img, clip_limit=0.02),
        image_rgb=image)
    io.imsave(fname=os.path.join(results_p_dir, f'01 02 - CLAHE.jpg'),
              arr=image)

    image_lab = rgb2lab(image)

    logging.info('Resizing')
    resize_factor = 8
    image_lab = resize(image, (int(image.shape[0] / resize_factor), (image.shape[1] / resize_factor)), anti_aliasing=True)
    io.imsave(fname=os.path.join(results_p_dir, f'01 03 - Resize.jpg'),
              arr=image_lab)

    brown_lab = np.array([29.01, 24.73, 39.85])
    blue_lab = np.array([36.72, 3.43, -23.77])
    white_lab = np.array([80.99, -1.56, -0.01])

    all_mask = segmentation(image_lab, 2, 2, 2, brown_lab, blue_lab, white_lab)



    """fig2 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(np.linspace(2, 6, 4), np.linspace(2, 6, 4), k, 60)
    plt.show()"""
