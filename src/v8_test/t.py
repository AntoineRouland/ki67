import datetime
import logging

import scipy.stats as scs
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io
from skimage import img_as_float
from skimage.color import rgb2lab, gray2rgb, lab2rgb
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian, try_all_threshold
from skimage.transform import resize
from skimage.morphology import disk, binary_opening
from mpl_toolkits import mplot3d
from math import sqrt, sin

from src.v1_color_patches.patch_classifier import PatchClassifier, pixelwise_closest_centroid, PROTOTYPES_Ki67_RGB
from src.v1_color_patches.color_segmentation import color_segmentation
from src.data_loader import sample_names, images, root_dir, FOLDER_EXPERIMENTS, references_names, references_paths, image_paths, originals_paths
from src.utils import apply_on_normalized_luminance, visualize_classification, colormap, outline_regions, crop
from src.v8_test.fct import weight, score_map_mse, create_se, create_mask_and_complementary, mse_cielab_on_image

def addi(brown, blue, white):
    return brown + blue + white


def test_on_all_images(param_name, param_initial_value, **dictionary):
    brown = dictionary.get('brown', 0)
    blue = dictionary.get('blue', 0)
    white = dictionary.get('white', 0)

    if param_name == 'brown':
        a = addi(param_initial_value, blue, white)
    elif param_name == 'blue':
        a = addi(brown, param_initial_value, white)
    elif param_name == 'white':
        a = addi(brown, blue, param_initial_value)
    return a


if __name__ == '__main__':
    history = []

    history.append({
        'parameter': 0,
        'value': 2,
        })

    history.append({
        'parameter': 5,
        'value': 9,
        })

    fichier = open("/home/uib/PycharmProjects/ki67/Results/history.txt", "w")
    fichier.write(f'{history})')
    fichier.close()




    """fig2 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(np.linspace(0, 20, 21), np.linspace(0, 20, 21), w, 160)
    plt.show()"""

