import datetime
import logging

import numpy as np
import os
import skimage.io as io
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.color import rgb2lab

from src.data_loader import root_dir, FOLDER_EXPERIMENTS, references_paths, references_names, originals_paths
from src.v8_test.segmentation import segmentation
from src.v8_test.validation import validation
from src.utils import apply_on_normalized_luminance

MAX_PATIENTS = 1
MAX_IMAGES_PER_PATIENT = 1
MAX_PATCHES_PER_IMAGE = 2


def test_on_all_images(radius):
    brown_lab = np.array([16.36, 110.89, 53.06])
    blue_lab = np.array([18.10, 50.58, -70.44])
    white_lab = np.array([53.15, -18.15, 1.20])

    k_list = []
    for j in range(nb_images):

        all_mask = segmentation(image_lab_list[j], radius[0], radius[1], radius[2],
                                brown_lab=brown_lab,
                                blue_lab=blue_lab,
                                white_lab=white_lab)

        k = validation(all_mask[0, :, :], all_mask[1, :, :], all_mask[2, :, :], ref_paths[j][0], ref_paths[j][1])
        print('k = ', k)

        k_list.append(k)

    print('k mean = ', np.mean(k_list))

    return np.mean(k_list)


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

    names = references_names()
    ref_paths = []
    ori_paths = []

    for i in range(len(names)):
        ref_paths.append(references_paths(names[i]))
        ori_paths.append(originals_paths(names[i]))

    nb_images = len(ref_paths)
    image_lab_list = []
    image_original_list = []

    for i in range(nb_images):

        image = io.imread(ori_paths[i][0])

        results_p_dir = os.path.join(results_dir, "image", f'{i}')
        os.makedirs(results_p_dir, exist_ok=True)
        io.imsave(fname=os.path.join(results_p_dir, '01 00 Original.jpg'),
                  arr=image)

        logging.info('Resizing')
        resize_factor = 8
        image_original_list.append(resize(image,
                                          (int(image.shape[0] / resize_factor), (image.shape[1] / resize_factor)),
                                   anti_aliasing=True))

        logging.info('Gaussian filter')
        image = apply_on_normalized_luminance(
            operation=lambda img: gaussian(img, sigma=2),
            image_rgb=image)
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
        image_lab = resize(image_lab, (int(image.shape[0] / resize_factor), (image.shape[1] / resize_factor)),
                           anti_aliasing=True)

        image_lab_list.append(image_lab)

    r = [[i, j, k] for i in range(2, 6) for j in range(2, 6) for k in range(2, 6)]
    k_mean_list = []

    for i in range(len(r)):
        k_mean_list.append(test_on_all_images(r[i]))

    k_max = np.max(k_mean_list)
    index = np.argmax(k_max)
    print(k_mean_list)
    print(f'k_max = {k_max} with {r[index]}')
