import datetime
import logging

import os
import skimage.io as io
from skimage import img_as_float
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.transform import resize

from src.patch_classifier import PatchClassifier, pixelwise_closest_centroid, PROTOTYPES_Ki67_RGB
from src.color_segmentation import color_segmentation
from src.data_loader import patient_names, images, root_dir, FOLDER_EXPERIMENTS
from src.utils import apply_on_normalized_luminance, visualize_classification, colormap, outline_regions, crop

MAX_PATIENTS = 1
MAX_IMAGES_PER_PATIENT = 1
MAX_PATCHES_PER_IMAGE = 2
RESIZE_IMAGES = (300, 300)  # None to deactivate

if __name__ == "__main__":
    execution_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_dir = root_dir(FOLDER_EXPERIMENTS, execution_id)
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    p_names = patient_names()
    for idx_p, p_name in enumerate(p_names[0:MAX_PATIENTS]):

        for idx_img, (path_image, image) in enumerate(images(patient_name=p_name, max_images=MAX_IMAGES_PER_PATIENT)):
            results_p_dir = os.path.join(results_dir, p_name, str(idx_img))
            os.makedirs(results_p_dir, exist_ok=True)
            logging.info(f'Processing: {p_name}-{idx_img}')

            if RESIZE_IMAGES:
                logging.info('Resizing image')
                image = resize(img_as_float(image), RESIZE_IMAGES + (3, ))
            io.imsave(fname=os.path.join(results_p_dir, '01 01 Original.jpg'),
                      arr=image)

            logging.info('Gaussian filter')
            image = apply_on_normalized_luminance(
                operation=gaussian,
                image_rgb=image)
            io.imsave(fname=os.path.join(results_p_dir, '01 02 Gaussian filter.jpg'),
                      arr=image)

            logging.info('CLAHE')
            image = apply_on_normalized_luminance(
                lambda img: equalize_adapthist(img, clip_limit=0.02),
                image_rgb=image)
            io.imsave(fname=os.path.join(results_p_dir, '01 03 CLAHE.jpg'),
                      arr=image)

            logging.info('Creating color segmentation')
            segmentation_idcs, segmentation_averaged = color_segmentation(image)
            io.imsave(fname=os.path.join(results_p_dir, '02 Segmentation idcs.jpg'),
                      arr=colormap(segmentation_idcs))
            io.imsave(fname=os.path.join(results_p_dir, '03 Segmentation averaged.jpg'),
                      arr=segmentation_averaged)

            logging.info('Classifying based on fixed centroids')
            classification, distance_maps = pixelwise_closest_centroid(
                image=segmentation_averaged,
                centroids_rgb=PROTOTYPES_Ki67_RGB)
            io.imsave(fname=os.path.join(results_p_dir, '04 Classification.png'),
                      arr=colormap(classification))
            io.imsave(fname=os.path.join(results_p_dir, '04 01 Positive.png'),
                      arr=colormap(classification == 0))
            io.imsave(fname=os.path.join(results_p_dir, '04 02 Negative.jpg'),
                      arr=colormap(classification == 1))
            io.imsave(fname=os.path.join(results_p_dir, '04 03 Background.jpg'),
                      arr=colormap(classification == 2))

            logging.info('Visualizing classification')
            classification_colored = visualize_classification(classification)
            io.imsave(fname=os.path.join(results_p_dir, '04 04 Classification colored.jpg'),
                      arr=classification_colored)

            logging.info('Creating some patches')
            c1 = PatchClassifier()
            patches = list(c1.patches(image=image, region_labels=segmentation_idcs))

            logging.info('Labelling patches')
            c1.label_patches(patches=patches, max_patches=MAX_PATCHES_PER_IMAGE)

            logging.info('Saving patches')
            c1.save_labelled_patches(training_label=execution_id)

            logging.info('Retrieving patches')
            c2 = PatchClassifier()
            c2.load_labelled_patches(training_label=execution_id)

            logging.info('Visualizing retrieved patches')
            for idx_patch, patch in enumerate(c2.labelled_patches):
                io.imsave(fname=os.path.join(results_p_dir, f'05 {idx_patch:02d} Patch image classification.jpg'),
                          arr=colormap(crop(image=segmentation_idcs, bounding_box=patch.bounding_box)))
                io.imsave(fname=os.path.join(results_p_dir, f'05 {idx_patch:02d} Patch image outlined.jpg'),
                          arr=outline_regions(patch.cropped_image(), patch.cropped_mask()))
                io.imsave(fname=os.path.join(results_p_dir, f'05 {idx_patch:02d} Patch image.jpg'),
                          arr=patch.cropped_image())
                io.imsave(fname=os.path.join(results_p_dir, f'05 {idx_patch:02d} Patch mask.jpg'),
                          arr=colormap(patch.cropped_mask()))
