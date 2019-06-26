import logging

import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt


from skimage.morphology import disk
from skimage.color import rgb2lab
from scipy.special import softmax

from src.utils import outline_regions, average_color
from src.v8_test.fct import score_map_mse, weight, weight_background, create_mask_and_complementary, create_se


def segmentation(image_lab, r_p, r_n, r_bg, brown_lab, blue_lab, white_lab):

    mask_p, mask_p_c = create_mask_and_complementary(r_p, (21, 21))
    mask_n, mask_n_c = create_mask_and_complementary(r_n, (21, 21))
    mask_b, mask_b_c = create_mask_and_complementary(r_bg, (21, 21))

    se_positive = create_se((21, 21, 3), brown_lab, white_lab, mask_p, mask_p_c)
    se_negative = create_se((21, 21, 3), blue_lab, white_lab, mask_n, mask_n_c)
    se_background = create_se((21, 21, 3), white_lab, white_lab, mask_b, mask_b_c)

    w_positive = weight(r_p, (21, 21))
    w_blue = weight(r_n, (21, 21))
    w_bg = weight_background(r_bg, (21, 21))

    logging.info('Score positive cell')
    score_positive = softmax(score_map_mse(image_lab, se_positive, w_positive, mask_p, mask_p_c))

    logging.info('Score negative cell')
    score_negative = softmax(score_map_mse(image_lab, se_negative, w_blue, mask_n, mask_n_c))

    logging.info('Score background')
    score_background = softmax(score_map_mse(image_lab, se_background, w_bg, mask_b, mask_b_c))

    all_score = np.zeros((3, score_positive.shape[0] - 20, score_positive.shape[1] - 20))
    all_score[0, :, :] = score_positive[10:score_positive.shape[0] - 10,
                                        10:score_positive.shape[1] - 10]
    all_score[1, :, :] = score_negative[10:score_positive.shape[0] - 10,
                                        10:score_positive.shape[1] - 10]
    all_score[2, :, :] = score_background[10:score_positive.shape[0] - 10,
                                          10:score_positive.shape[1] - 10]

    all_mask = np.zeros((3, score_positive.shape[0] - 20, score_positive.shape[1] - 20), int)

    for i in range(all_score.shape[1]):
        for j in range(all_score.shape[2]):
            index = np.argmin(all_score[:, i, j])
            all_mask[index, i, j] = 1

    return all_mask
