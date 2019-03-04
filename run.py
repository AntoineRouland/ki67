import datetime
import logging
import os
import skimage

from data_loader import patient_names, images, num_images

if __name__ == "__main__":
    results_dir = os.path.join('Results', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    for idx_p, p_name in enumerate(patient_names()):

        for idx_img, (path_image, image) in enumerate(images(patient_name=p_name)):
            results_p_dir = os.path.join(results_dir, p_name, str(idx_img))
            os.makedirs(results_p_dir, exist_ok=True)
            logging.info(f'Processing: {p_name}-{idx_img}')

            skimage.io.imsave(fname=os.path.join(results_p_dir, f' Original.jpg'),
                              arr=image)
