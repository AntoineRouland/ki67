
import numpy as np
from skimage.morphology import disk


def mse_cielab(image, w, d, dc, color_foreground, color_background):
    dl = d * (image[:, :, 0] - color_foreground[0]) + dc * (image[:, :, 0] - color_background[0])
    db = d * (image[:, :, 1] - color_foreground[1]) + dc * (image[:, :, 1] - color_background[1])
    da = d * (image[:, :, 2] - color_foreground[2]) + dc * (image[:, :, 2] - color_background[2])
    distance = np.sqrt(dl**2 + da**2 + db**2)
    error = w * distance
    return error


def score_map_mse(image, se, w, d, dc):
    color_foreground = se[int(se.shape[0] / 2), int(se.shape[1] / 2)]
    color_background = se[0, 0]
    m = np.zeros(image.shape[0:2])
    for i in range(15, image.shape[0] - 15):
        for j in range(15, image.shape[1] - 15):
            m[i, j] = np.mean(mse_cielab(image[i - (int(se.shape[0] / 2)):i + (int(se.shape[0] / 2))+1,
                              j - (int(se.shape[0] / 2)):j + (int(se.shape[0] / 2))+1, :],
                              w, d, dc, color_foreground, color_background))
    return m


def weight(r_center, shape):
    d1 = np.zeros(shape)
    d1[int(shape[0]/2) - r_center:int(shape[0]/2) + r_center + 1,
       int(shape[0]/2) - r_center:int(shape[0]/2) + r_center + 1] = disk(r_center)
    d2 = np.zeros(shape)
    d2[1:shape[0]-1, 1:shape[0]-1] = disk(int(shape[0]/2)-1)
    d2 = (np.ones(shape) - d2)*0.01
    return d1 + d2


def weight_background(r_center, shape):
    d1 = np.zeros(shape)
    d1[int(shape[0]/2) - r_center:int(shape[0]/2) + r_center + 1,
       int(shape[0]/2) - r_center:int(shape[0]/2) + r_center + 1] = disk(r_center)
    return d1