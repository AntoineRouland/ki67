import datetime
import logging
from math import ceil

import os
import skimage.io as io
from skimage import img_as_float
from skimage.transform import resize

from src.data_loader import sample_names, images, root_dir, FOLDER_EXPERIMENTS, get_expert_Ki67


MAX_PATIENTS = 1000
MAX_IMAGES_PER_PATIENT = 5
MAX_PATCHES_PER_IMAGE = 2

SCALE = None  # None to deactivate
GAUSSIAN_FILTER_SD = 2
CLUSTERING_NUM_CENTROIDS = 4
RADIUS_FILL_HOLES = 5
WIDTH_REMOVE_THIN_STRUCTURES = 12

if __name__ == "__main__":
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_dir = root_dir(FOLDER_EXPERIMENTS(version=6), execution_id)
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    p_names = sample_names()
    for idx_p, p_name in enumerate(p_names[0:MAX_PATIENTS]):



        for idx_img, (path_image, original_image) in enumerate(images(patient_name=p_name, max_images=MAX_IMAGES_PER_PATIENT)):

            Ki67 = get_expert_Ki67(p_name)

            if SCALE:
                logging.info('Resizing image')
                sz = [ceil(d*SCALE) for d in original_image.shape[:2]] + [3]
                original_image = resize(img_as_float(original_image), sz)
            io.imsave(fname=os.path.join(results_dir, f'{idx_p:02d} {p_name} - Image {idx_img:02d} - Ki67: {Ki67}.jpg'),
                      arr=original_image)
            image = original_image

