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
from src.v8_test.fct import score_map_mse, weight


def validation(positive, negative, background):

    mask_positive = io.imread("/home/uib/PycharmProjects/ki67/labelbox_export-master/output/dataset 2019-49-06 10-49-42/1309620 -A2-27/mask_positive.png")
    mask_negative = io.imread("/home/uib/PycharmProjects/ki67/labelbox_export-master/output/dataset 2019-49-06 10-49-42/1309620 -A2-27/mask_negative.png")
    mask_positive = (mask_positive > 128) * 1
    mask_negative = (mask_negative > 128) * 1
    mask_negative = mask_negative - (mask_negative & mask_positive)
    mask_background = np.ones(mask_positive.shape, int) - (mask_positive | mask_negative)

    resize_factor = 8
    mask_positive = (resize(mask_positive,
                            (int(mask_positive.shape[0] / resize_factor), (mask_positive.shape[1] / resize_factor)),
                            anti_aliasing=False) > 5.43*10**(-20)) * 1
    mask_negative = (resize(mask_negative,
                            (int(mask_negative.shape[0] / resize_factor), (mask_negative.shape[1] / resize_factor)),
                            anti_aliasing=False) > 5.43*10**(-20)) * 1
    mask_background = (resize(mask_background,
                              (int(mask_background.shape[0] / resize_factor),
                               (mask_background.shape[1] / resize_factor)),
                              anti_aliasing=False) > 5.43*10**(-20)) * 1

    a, b = mask_positive.shape
    mask_positive = mask_positive[15:a - 15, 15:b - 15]
    mask_negative = mask_negative[15:a - 15, 15:b - 15]
    mask_background = mask_background[15:a - 15, 15:b - 15]

    references = [mask_positive, mask_negative, mask_background]
    results = [positive, negative, mask_background]
    mij = np.zeros((3, 3), int)

    for i in range(3):
        for j in range(3):
            mij[i, j] = np.sum(references[i] & results[j])

    success_rate = (mij[0, 0] + mij[1, 1] + mij[2, 2]) / np.sum(mij)
    return success_rate
