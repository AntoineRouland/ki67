import datetime
import logging

import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt

from skimage.morphology import disk
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.color import rgb2lab
from scipy.special import softmax

from src.data_loader import root_dir, FOLDER_EXPERIMENTS
from src.utils import apply_on_normalized_luminance, outline_regions
from src.v8_test.fct import score_map_mse, weight, weight_background


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

    results_p_dir = os.path.join(results_dir, "image", str(1))
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

    logging.info('Resizing')
    resize_factor = 8
    image = resize(image, (int(image.shape[0] / resize_factor), (image.shape[1] / resize_factor)), anti_aliasing=True)
    io.imsave(fname=os.path.join(results_p_dir, f'01 03 - Resize.jpg'),
              arr=image)

    image_lab = rgb2lab(image)

    d1 = np.zeros((31, 31))
    r1 = 3
    d1[15 - r1:15 + r1 + 1, 15 - r1:15 + r1 + 1] = disk(r1)
    dc1 = np.ones((31, 31)) - d1

    d2 = np.zeros((31, 31))
    r2 = 2
    d2[15 - r2:15 + r2 + 1, 15 - r2:15 + r2 + 1] = disk(r2)
    dc2 = np.ones((31, 31)) - d2

    d3 = np.zeros((31, 31))
    r3 = 2
    d3[15 - r3:15 + r3 + 1, 15 - r3:15 + r3 + 1] = disk(r3)
    dc3 = np.ones((31, 31)) - d3

    brown_lab = np.array([29.01, 24.73, 39.85])
    blue_lab = np.array([36.72, 3.43, -23.77])
    white_lab = np.array([80.99, -1.56, -0.01])
    SE_positive = np.zeros((31, 31, 3))
    for i in range(d1.shape[0]):
        for j in range(d1.shape[1]):
            if d1[i, j] == 1:
                SE_positive[i, j, :] = brown_lab
            else:
                SE_positive[i, j, :] = white_lab

    SE_negative = np.zeros((31, 31, 3))
    for i in range(d2.shape[0]):
        for j in range(d2.shape[1]):
            if d2[i, j] == 1:
                SE_negative[i, j, :] = blue_lab
            else:
                SE_negative[i, j, :] = white_lab

    SE_background = np.zeros((31, 31, 3))
    for i in range(d3.shape[0]):
        for j in range(d3.shape[1]):
            if d3[i, j] == 1:
                SE_background[i, j, :] = white_lab

    w_positive = weight(r1, (31, 31))
    w_blue = weight(r2, (31, 31))
    w_bg = weight_background(r3, (31, 31))

    logging.info('Score positive cell')
    score_positive = softmax(score_map_mse(image_lab, SE_positive, w_positive, d1, dc1))
    # score_positive = score_positive / np.max(score_positive)
    io.imsave(fname=os.path.join(results_p_dir, f'01 04 - score positive.jpg'),
              arr=score_positive)

    logging.info('Score negative cell')
    score_negative = softmax(score_map_mse(image_lab, SE_negative, w_blue, d2, dc2))
    # score_negative = score_negative / np.max(score_negative)
    io.imsave(fname=os.path.join(results_p_dir, f'01 05 - score negative.jpg'),
              arr=score_negative)

    logging.info('Score background')
    score_background = softmax(score_map_mse(image_lab, SE_background, w_bg, d2, dc2))
    # score_background = score_background / np.max(score_background)
    io.imsave(fname=os.path.join(results_p_dir, f'01 06 - score background.jpg'),
              arr=score_background)

    all_score = np.zeros((3, score_positive.shape[0] - 30, score_positive.shape[1] - 30))
    all_score[0, :, :] = score_positive[15:score_positive.shape[0] - 15,
                                        15:score_positive.shape[1] - 15]
    all_score[1, :, :] = score_negative[15:score_positive.shape[0] - 15,
                                        15:score_positive.shape[1] - 15]
    all_score[2, :, :] = score_background[15:score_positive.shape[0] - 15,
                                          15:score_positive.shape[1] - 15]

    plt.subplot(221)
    plt.imshow(score_positive)
    plt.subplot(222)
    plt.imshow(score_negative)
    plt.subplot(223)
    plt.imshow(score_background)
    plt.show()

    all_mask = np.zeros((3, score_positive.shape[0] - 30, score_positive.shape[1] - 30), int)

    for i in range(all_score.shape[1]):
        for j in range(all_score.shape[2]):
            index = np.argmin(all_score[:, i, j])
            all_mask[index, i, j] = 1

    """io.imsave(fname=os.path.join(results_p_dir, f'01 08 - mask positive.jpg'),
              arr=all_mask[0, :, :])
    io.imsave(fname=os.path.join(results_p_dir, f'01 07 - mask negative.jpg'),
              arr=all_mask[1, :, :])
    io.imsave(fname=os.path.join(results_p_dir, f'01 09 - mask background.jpg'),
              arr=all_mask[2, :, :])"""

    regions_positive = outline_regions(image[15:score_positive.shape[0] - 15, 15:score_positive.shape[1] - 15],
                                       all_mask[0, :, :])
    io.imsave(fname=os.path.join(results_p_dir, f'01 10 - regions positive.jpg'),
              arr=regions_positive)

    regions_negative = outline_regions(image[15:score_positive.shape[0] - 15, 15:score_positive.shape[1] - 15],
                                       all_mask[1, :, :])
    io.imsave(fname=os.path.join(results_p_dir, f'01 11 - regions negative.jpg'),
              arr=regions_negative)

    regions_background = outline_regions(image[15:score_positive.shape[0] - 15, 15:score_positive.shape[1] - 15],
                                         all_mask[2, :, :])
    io.imsave(fname=os.path.join(results_p_dir, f'01 12 - regions background.jpg'),
              arr=regions_background)
