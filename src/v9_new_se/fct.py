import numpy as np
import matplotlib.pyplot as plt
import os


def mse_cielab_on_image(image_lab, color_lab):
    """ This function compute the mean square error between a color and each pixel of an image.

    :param image_lab: an image in the color lab space
    :param color_lab: a color in the color lab space
    :return: a matrix for error with the same size of image_lab
    """

    dl = image_lab[:, :, 0] - color_lab[0]
    da = image_lab[:, :, 1] - color_lab[1]
    db = image_lab[:, :, 2] - color_lab[2]
    return np.sqrt(dl**2 + da**2 + db**2)


def score_map_mse(image_lab, color, w):
    """ This function compute the degree of fitting of a color in an image, and return the information as a score map
    with the degree of fitting for each position in the image.

    :param image_lab: an image in the color lab space
    :param color: a color in the color lab space
    :param w: a weight matrix, which give the degree of importance of the neighbourhood of a pixel
    :return: a score map for each pixel, expect the ones on the sides in order to avoid noise
    """
    u = int(w.shape[0] / 2)
    error_matrix_color_foreground = mse_cielab_on_image(image_lab, color)
    m = np.zeros(image_lab.shape[0:2])
    for i in range(u, image_lab.shape[0] - u):
        for j in range(u, image_lab.shape[1] - u):
            m[i, j] = np.mean(w * error_matrix_color_foreground[i - u:i + u + 1, j - u:j + u + 1])
    return m


def gaus2d(x, y, mx=0, my=0, sx=1, sy=1):
    """ Return a 2D gaussian.

    :param x: interval of x values
    :param y: interval of y values
    :param mx: expected value on x
    :param my: expected value on y
    :param sx: variance on x
    :param sy: variance on y
    :return: a 2D matrix
    """
    g = np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
    return g / np.max(g)


def weight(length, sigma):
    """ This function return a weight matrix with a 2 dimensional gaussian centered in the center of the matrix.

    :param length: length of the sides of the square matrix
    :param sigma: variance of the gaussian
    :return: a square matrix
    """
    x = np.linspace(0, length-1, length)
    y = np.linspace(0, length-1, length)
    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D
    z = gaus2d(x, y, mx=int(length/2), my=int(length/2), sx=sigma, sy=sigma)
    return z


def plot_kappa_score(k_history, nb_image, results_dir):
    """ This function saves the Cohen's kappa score for every image and the average score in the specified directory.

    :param k_history: list of lists with scores for every images and the average score
    :param nb_image: the number of images
    :param results_dir: the directory where the figures will be save
    :return: None
    """

    for i in range(nb_image):
        fig = plt.figure()
        ax = plt.axes()
        plt.plot(np.linspace(1, len(k_history[i]), len(k_history[i])), k_history[i])
        plt.title(f'cohen\'s kappa score for image{i}')
        ax.set(xlabel='iterations', ylabel='cohen\'s kappa score')
        plt.ylim((0, 1))
        plt.savefig(fname=os.path.join(results_dir, f'cohen\'s kappa score for image {i}.jpg'))

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(np.linspace(1, len(k_history[-1]), len(k_history[-1])), k_history[-1])
    plt.title(f'average cohen\'s kappa score')
    ax.set(xlabel='iterations', ylabel='cohen\'s kappa score')
    plt.ylim((0, 1))
    plt.savefig(fname=os.path.join(results_dir, f'average cohen\'s kappa score.jpg'))


def ki_67_percentage(mask_positive, mask_negative):
    """ This function computes the percentage of positive cells in and image based on the masks obtained previously.

    :param mask_positive: mask for the positives cells on the image
    :param mask_negative: mask for the negatives cells on the image
    :return: a float
    """
    total_area = mask_positive.shape[0] * mask_positive.shape[1]
    area_positive = np.sum(mask_positive)
    visible_area_negative = np.sum(mask_negative)
    visible_area_negative_percentage = visible_area_negative / (total_area - area_positive)
    hidden_area_negative = visible_area_negative_percentage * area_positive
    area_negative = visible_area_negative + hidden_area_negative
    return area_positive / (area_positive + area_negative)
