import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io
from skimage import img_as_float
from skimage.color import rgb2lab, gray2rgb
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian, threshold_isodata
from skimage.transform import resize
from skimage.morphology import disk
from mpl_toolkits import mplot3d

from src.v1_color_patches.patch_classifier import PatchClassifier, pixelwise_closest_centroid, PROTOTYPES_Ki67_RGB
from src.v1_color_patches.color_segmentation import color_segmentation
from src.data_loader import sample_names, images, root_dir, FOLDER_EXPERIMENTS
from src.utils import apply_on_normalized_luminance, visualize_classification, colormap, outline_regions, crop
from src.v8_test.fct import score_map_mse, weight, weight_background
from src.v8_test.validation import validation


if __name__ == "__main__":

    image = io.imread("/home/uib/PycharmProjects/ki67/test/01 02 - CLAHE.jpg")
    resize_factor = 8
    image = resize(image, (int(image.shape[0] / resize_factor), (image.shape[1] / resize_factor)), anti_aliasing=True)

    image_lab = rgb2lab(image)
    success_rate = np.zeros((8,))

    r1 = 2
    r3 = 3

    for r2 in range(2, 10):

        d1 = np.zeros((31, 31))
        d1[15 - r1:15 + r1 + 1, 15 - r1:15 + r1 + 1] = disk(r1)
        dc1 = np.ones((31, 31)) - d1

        d2 = np.zeros((31, 31))
        d2[15 - r2:15 + r2 + 1, 15 - r2:15 + r2 + 1] = disk(r2)
        dc2 = np.ones((31, 31)) - d2

        d3 = np.zeros((31, 31))
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

        score_positive = score_map_mse(image_lab, SE_positive, w_positive, d1, dc1)
        """score_positive = score_positive / np.max(score_positive)"""
        score_negative = score_map_mse(image_lab, SE_negative, w_blue, d2, dc2)
        score_negative = score_negative / np.max(score_negative)
        score_background = score_map_mse(image_lab, SE_background, w_bg, d2, dc2)
        score_background = score_background / np.max(score_background)

        """thresh_negative = threshold_isodata(score_negative[15:score_negative.shape[0] - 15,
                                                15:score_negative.shape[1] - 15])
            mask_negative = score_negative[15:score_negative.shape[0] - 15, 15:score_negative.shape[1] - 15] < thresh_negative
            mask_negative = mask_negative * 1

            thresh_positive = threshold_isodata(score_positive[15:score_positive.shape[0] - 15,
                                                15:score_positive.shape[1] - 15])
            mask_positive = score_positive[15:score_positive.shape[0] - 15, 15:score_positive.shape[1] - 15] < thresh_positive
            mask_positive = mask_positive * 1

            thresh_background = threshold_isodata(score_background[15:score_background.shape[0] - 15,
                                                  15:score_background.shape[1] - 15])
            mask_background = score_background[15:score_background.shape[0] - 15, 15:score_background.shape[1] - 15] < thresh_background
            mask_background = mask_background * 1 """

        all_score = np.zeros((3, score_positive.shape[0] - 30, score_positive.shape[1] - 30))
        all_score[0, :, :] = score_positive[15:score_positive.shape[0] - 15,
                                 15:score_positive.shape[1] - 15]
        all_score[1, :, :] = score_negative[15:score_positive.shape[0] - 15,
                                 15:score_positive.shape[1] - 15]
        all_score[2, :, :] = score_background[15:score_positive.shape[0] - 15,
                                 15:score_positive.shape[1] - 15]

        all_mask = np.zeros((3, score_positive.shape[0] - 30, score_positive.shape[1] - 30), int)

        for i in range(all_score.shape[1]):
            for j in range(all_score.shape[2]):
                index = np.argmin(all_score[:, i, j])
                all_mask[index, i, j] = 1
        success_rate[r2-2] = validation(all_mask[0, :, :], all_mask[1, :, :], all_mask[2, :, :])

        print(success_rate[r2-2])

    print(success_rate)

    plt.plot(np.linspace(2, 9, 8), success_rate)
    plt.show()



