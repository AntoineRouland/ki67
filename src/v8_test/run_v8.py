import datetime
import logging

import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.color import rgb2lab

from src.data_loader import root_dir, FOLDER_EXPERIMENTS, sample_names, images
from src.v8_test.segmentation import segmentation
from src.utils import apply_on_normalized_luminance
from src.utils import outline_regions
from src.v8_test.fct import ki_67_percentage

MAX_PATIENTS = 3
MAX_IMAGES_PER_PATIENT = 25


if __name__ == "__main__":
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_dir = root_dir(FOLDER_EXPERIMENTS(version=8), execution_id)
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    file = open(os.path.join(results_dir, "percentage of ki-67.txt"), "w")

    percentage_list = [[] for i in range(MAX_PATIENTS)]

    p_names = sample_names()
    for idx_p, p_name in enumerate(p_names[0:MAX_PATIENTS]):

        for idx_img, (path_image, original_image) in enumerate(images(patient_name=p_name,
                                                                      max_images=MAX_IMAGES_PER_PATIENT)):

            results_p_dir = os.path.join(results_dir, p_name, str(idx_img))
            os.makedirs(results_p_dir, exist_ok=True)
            logging.info(f'Processing: {p_name}-{idx_img}')

            logging.info('Resizing')
            resize_factor = 8
            resized_original = (resize(original_image,
                                       (int(original_image.shape[0] / resize_factor),
                                        (original_image.shape[1] / resize_factor)),
                                       anti_aliasing=True))

            logging.info('Gaussian filter')
            image = apply_on_normalized_luminance(
                operation=lambda img: gaussian(img, sigma=2),
                image_rgb=original_image)
            io.imsave(fname=os.path.join(results_p_dir, f'01 01 - Gaussian filter.jpg'),
                      arr=image)

            logging.info('CLAHE')
            image = apply_on_normalized_luminance(
                lambda img: equalize_adapthist(img, clip_limit=0.02),
                image_rgb=image)
            io.imsave(fname=os.path.join(results_p_dir, f'01 02 - CLAHE.jpg'),
                      arr=image)

            image_lab = rgb2lab(image)

            logging.info('Resizing')
            resize_factor = 8
            image_lab = resize(image_lab, (int(image.shape[0] / resize_factor), (image.shape[1] / resize_factor)),
                               anti_aliasing=True)

            all_mask = segmentation(image_lab, 2, 2, 2,
                                    brown_lab=np.array([16.36, 110.89, 53.06]),
                                    blue_lab=np.array([18.10, 50.58, -70.44]),
                                    white_lab=np.array([53.15, -18.15, 1.20]))

            regions_positive = outline_regions(resized_original[10:resized_original.shape[0] - 10,
                                               10:resized_original.shape[1] - 10], all_mask[0, :, :])
            io.imsave(fname=os.path.join(results_p_dir, f'regions positive.jpg'),
                      arr=regions_positive)
            regions_negative = outline_regions(resized_original[10:resized_original.shape[0] - 10,
                                               10:resized_original.shape[1] - 10], all_mask[1, :, :])
            io.imsave(fname=os.path.join(results_p_dir, f'regions negative.jpg'),
                      arr=regions_negative)
            regions_background = outline_regions(resized_original[10:resized_original.shape[0] - 10,
                                                 10:resized_original.shape[1] - 10], all_mask[2, :, :])
            io.imsave(fname=os.path.join(results_p_dir, f'regions background.jpg'),
                      arr=regions_background)

            p = ki_67_percentage(all_mask[0, :, :], all_mask[1, :, :])
            print(p)
            percentage_list[idx_p].append(p)

        m = np.mean(percentage_list[idx_p])
        percentage_list[idx_p].append(m)
        file.write(f'The average percentage of cells marked by the ki-67 antigen for the patient {p_name} is {m} \n')

    file.close()

    plt.xlim(10, 40)
    plt.xlabel('Experts Values')
    plt.ylabel('Method Outputs')

    for i in range(len(percentage_list)):
        if i == 0:
            v = 30
        elif i == 1:
            v = 20
        elif i == 2:
            v = 15
        for j in range(len(percentage_list[i])):
            plt.plot(v, percentage_list[i][j], 'b+')
            plt.text(v+0.3, percentage_list[i][j], f'{j}')
        plt.plot(v, np.mean(percentage_list[i]), 'r+')
        p50 = np.percentile(percentage_list[i], 50)
        p75 = np.percentile(percentage_list[i], 75)
        plt.plot(v, p50, 'g+')
        plt.text(v+0.7, p50, '50th percentile')
        plt.plot(v, p75, 'g+')
        plt.text(v+0.7, p75, '75th percentile')

    plt.show()
