import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io
from skimage import img_as_float
from skimage.color import rgb2lab, gray2rgb, lab2rgb, rgb2gray
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian, try_all_threshold, threshold_isodata
from skimage.transform import resize
from skimage.morphology import disk
from mpl_toolkits import mplot3d
from scipy.special import softmax

from src.v1_color_patches.patch_classifier import PatchClassifier, pixelwise_closest_centroid, PROTOTYPES_Ki67_RGB
from src.v1_color_patches.color_segmentation import color_segmentation
from src.data_loader import sample_names, images, root_dir, FOLDER_EXPERIMENTS
from src.utils import apply_on_normalized_luminance, visualize_classification, colormap, outline_regions, crop
from src.v8_test.fct import score_map_mse, weight, weight_background
from src.v8_test.validation import validation


MAX_PATIENTS = 1
MAX_IMAGES_PER_PATIENT = 1
MAX_PATCHES_PER_IMAGE = 2


if __name__ == "__main__":

    image = io.imread("/home/uib/PycharmProjects/ki67/Results/Experiments v8/oui/image/1/01 02 - CLAHE.jpg")

    resize_factor = 8
    image = resize(image, (int(image.shape[0] / resize_factor), (image.shape[1] / resize_factor)), anti_aliasing=True)

    image_lab = rgb2lab(image)

    r1 = 7
    r2 = 2
    d1 = np.zeros((31, 31))
    d1[15 - r1:15 + r1 + 1, 15 - r1:15 + r1 + 1] = disk(r1)
    dc1 = np.ones((31, 31)) - d1

    d2 = np.zeros((11, 11))
    d2[5 - r2:5 + r2 + 1, 5 - r2:5 + r2 + 1] = disk(r2)
    dc2 = np.ones((11, 11)) - d2

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

    SE_negative = np.zeros((11, 11, 3))
    for i in range(d2.shape[0]):
        for j in range(d2.shape[1]):
            if d2[i, j] == 1:
                SE_negative[i, j, :] = blue_lab
            else:
                SE_negative[i, j, :] = white_lab

    SE_background = np.zeros((11, 11, 3))
    for i in range(d2.shape[0]):
        for j in range(d2.shape[1]):
            if d2[i, j] == 1:
                SE_background[i, j, :] = white_lab

    w_positive = weight(r1, (31, 31))
    w_blue = weight(r2, (11, 11))
    w_bg = weight_background(r2, (11, 11))

    score_positive = score_map_mse(image_lab, SE_positive, w_positive, d1, dc1)
    score_negative = score_map_mse(image_lab, SE_negative, w_blue, d2, dc2)
    score_background = score_map_mse(image_lab, SE_background, w_bg, d2, dc2)

    all_score = np.zeros((3, score_positive.shape[0]-30, score_positive.shape[1]-30))
    all_score[0, :, :] = score_positive[15:score_positive.shape[0] - 15,
                                        15:score_positive.shape[1] - 15]
    all_score[1, :, :] = score_negative[15:score_positive.shape[0] - 15,
                                        15:score_positive.shape[1] - 15]
    all_score[2, :, :] = score_background[15:score_positive.shape[0] - 15,
                                          15:score_positive.shape[1] - 15]

    all_mask = np.zeros((3, score_positive.shape[0]-30, score_positive.shape[1]-30))

    for i in range(all_score.shape[1]):
        for j in range(all_score.shape[2]):
            index = np.argmax(all_score[:, i, j])
            all_mask[index, i, j] = 1



