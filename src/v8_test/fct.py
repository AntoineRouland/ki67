import numpy as np
from skimage.morphology import disk
import matplotlib.pyplot as plt


def gauss_2d(mu, sigma, length):
    x, y = np.meshgrid(np.linspace(-1, 1, length), np.linspace(-1, 1, length))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g


def mse_cielab_on_image(image_lab, color_lab):
    dl = image_lab[:, :, 0] - color_lab[0]
    da = image_lab[:, :, 1] - color_lab[1]
    db = image_lab[:, :, 2] - color_lab[2]
    return np.sqrt(dl**2 + da**2 + db**2)


def score_map_mse(image_lab, se, w, mask, mask_c):
    u = int(se.shape[0] / 2)
    color_foreground = se[u, u, :]
    color_background = se[0, 0, :]
    error_matrix_color_foreground = mse_cielab_on_image(image_lab, color_foreground)
    error_matrix_color_background = mse_cielab_on_image(image_lab, color_background)
    m = np.zeros(image_lab.shape[0:2])
    for i in range(u, image_lab.shape[0] - u):
        for j in range(u, image_lab.shape[1] - u):
            m[i, j] = np.mean(w * (error_matrix_color_foreground[i - u:i + u + 1, j - u:j + u + 1] * mask +
                                   error_matrix_color_background[i - u:i + u + 1, j - u:j + u + 1] * mask_c))
    return m


def weight(r_center, shape):
    d1 = np.zeros(shape)
    d1[int(shape[0]/2) - r_center:int(shape[0]/2) + r_center + 1,
       int(shape[0]/2) - r_center:int(shape[0]/2) + r_center + 1] = disk(r_center)
    d2 = np.zeros(shape)
    d2[0:shape[0], 0:shape[0]] = disk(int(shape[0]/2))
    d2 = (np.ones(shape) - d2) * 0.01
    return d1 + d2


def weight_background(r_center, shape):
    d1 = np.zeros(shape)
    d1[int(shape[0]/2) - r_center:int(shape[0]/2) + r_center + 1,
       int(shape[0]/2) - r_center:int(shape[0]/2) + r_center + 1] = disk(r_center)
    return d1


def create_mask_and_complementary(r_center, shape):
    d = np.zeros(shape, int)
    d[int(shape[0]/2) - r_center:int(shape[0]/2) + r_center + 1,
      int(shape[0]/2) - r_center:int(shape[0]/2) + r_center + 1] = disk(r_center)
    return d, np.ones(shape) - d


def create_se(shape, color_foreground, color_background, mask, mask_c):
    se = np.zeros(shape)
    se[:, :, 0] = mask * color_foreground[0] + mask_c * color_background[0]
    se[:, :, 1] = mask * color_foreground[1] + mask_c * color_background[1]
    se[:, :, 2] = mask * color_foreground[2] + mask_c * color_background[2]
    return se
