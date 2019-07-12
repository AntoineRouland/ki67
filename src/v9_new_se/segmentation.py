import logging

import numpy as np

from scipy.special import softmax

from src.v9_new_se.fct import score_map_mse, weight


def segmentation(image_lab, brown_lab, blue_lab, white_lab, sigma):

    w_positive = weight(11, sigma)
    w_blue = weight(11, sigma)
    w_bg = weight(11, sigma)

    logging.info('Score positive cell')
    score_positive = softmax(score_map_mse(image_lab, brown_lab, w_positive))

    logging.info('Score negative cell')
    score_negative = softmax(score_map_mse(image_lab, blue_lab, w_blue))

    logging.info('Score background')
    score_background = softmax(score_map_mse(image_lab, white_lab, w_bg))

    all_score = np.zeros((3, score_positive.shape[0] - 10, score_positive.shape[1] - 10))
    all_score[0, :, :] = score_positive[5:score_positive.shape[0] - 5,
                                        5:score_positive.shape[1] - 5]
    all_score[1, :, :] = score_negative[5:score_positive.shape[0] - 5,
                                        5:score_positive.shape[1] - 5]
    all_score[2, :, :] = score_background[5:score_positive.shape[0] - 5,
                                          5:score_positive.shape[1] - 5]

    all_mask = np.zeros((3, score_positive.shape[0] - 10, score_positive.shape[1] - 10), int)

    for i in range(all_score.shape[1]):
        for j in range(all_score.shape[2]):
            index = np.argmin(all_score[:, i, j])
            all_mask[index, i, j] = 1

    return all_mask
