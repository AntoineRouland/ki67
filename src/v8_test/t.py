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
from src.data_loader import sample_names, images, root_dir, FOLDER_EXPERIMENTS
from src.utils import apply_on_normalized_luminance, visualize_classification, colormap, outline_regions, crop
from src.v8_test.fct import weight, score_map_mse
from scipy.special import softmax

if __name__ == "__main__":

    a = np.array([[2, 6, 3], [4, 9, 6]])
    print(a)
    print(softmax(a))



    """fig2 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(np.linspace(2, 10, 8), np.linspace(2, 10, 8), success_rate, 160)
    plt.show()"""

