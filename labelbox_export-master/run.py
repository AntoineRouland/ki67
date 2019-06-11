from datetime import datetime
import json
import logging
import os

from skimage import io
from skimage.morphology import binary_opening, disk

log = logging.getLogger(__name__)

filename = 'export-2019-06-05T15_23_07.534Z.json'
dataset_dir = os.path.join('output', 'dataset ' + datetime.now().strftime('%Y-%M-%d %H-%M-%S'))


def read_data(fname):
    with open(fname, 'r') as f:
        return json.load(f)


def save_masks(name, original, mask_positive, mask_negative):
    directory = os.path.join(dataset_dir, name)
    os.makedirs(directory)
    io.imsave(os.path.join(directory, 'original.jpg'), original)
    io.imsave(os.path.join(directory, 'mask_positive.png'), mask_positive)
    io.imsave(os.path.join(directory, 'mask_negative.png'), mask_negative)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

    log.info('Read data from .json file')
    data = read_data(filename)
    data = [sample for sample in data if 'Masks' in sample]     # Restrict to samples that contain masks.
    for sample in data:
        name = os.path.splitext(sample['External ID'])[0]

        log.info(f'[Sample {name}] Download image and masks')
        original = io.imread(sample['Labeled Data'])
        mask_positive = io.imread(sample['Masks']['Positive'])
        mask_negative = io.imread(sample['Masks']['Negative'])

        log.info(f'[Sample {name}] Process')
        mask_positive = binary_opening(mask_positive[:, :, 0], disk(14)) * 1
        mask_negative = binary_opening(mask_negative[:, :, 0], disk(14)) * 1

        log.info(f'[Sample {name}] Save masks')
        save_masks(name, original, mask_positive, mask_negative)
