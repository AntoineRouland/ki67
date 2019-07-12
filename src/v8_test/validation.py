import numpy as np
import skimage.io as io

from skimage.transform import resize
from sklearn.metrics import cohen_kappa_score


def validation(positive, negative, background, mask_positive_path, mask_negative_path):

    mask_positive = io.imread(mask_positive_path)
    mask_negative = io.imread(mask_negative_path)
    resize_factor = 8
    mask_positive = (resize(mask_positive,
                            (int(mask_positive.shape[0] / resize_factor), (mask_positive.shape[1] / resize_factor)),
                            anti_aliasing=False) > 5.43 * 10 ** (-20)) * 1
    mask_negative = (resize(mask_negative,
                            (int(mask_negative.shape[0] / resize_factor), (mask_negative.shape[1] / resize_factor)),
                            anti_aliasing=False) > 5.43 * 10 ** (-20)) * 1

    mask_negative = mask_negative - (mask_negative & mask_positive)
    mask_background = np.ones(mask_positive.shape, int) - (mask_positive | mask_negative)

    a, b = mask_positive.shape
    mask_positive = mask_positive[10:a - 10, 10:b - 10]
    mask_negative = mask_negative[10:a - 10, 10:b - 10]
    mask_background = mask_background[10:a - 10, 10:b - 10]

    references = 3*mask_positive + 2*mask_negative + mask_background
    references = references.reshape((-1, 1))
    results = 3*positive + 2*negative + background
    results = results.reshape((-1, 1))

    return cohen_kappa_score(references, results)
