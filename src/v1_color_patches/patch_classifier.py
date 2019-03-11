import itertools
import json
import os
import random
import string

import numpy as np
from easygui import buttonbox
from skimage import img_as_float
from skimage.io import imsave, imread
from skimage.transform import resize

from src.data_loader import root_dir, FOLDER_LABELLED_DATA, FOLDER_TEMP_DATA
from src.utils import outline_regions, crop, hash_np

PROTOTYPES_Ki67_RGB = {
    'Positive': [(67, 30, 21), (108, 95, 89), (105, 86, 71), (82, 35, 7),
                 (103, 62, 30), (61, 45, 55), (90, 59, 39), (100, 63, 47)],
    'Negative': [(111, 113, 125), (105, 102, 119), (101, 100, 114), (134, 133, 139),
                 (119, 116, 133), (108, 108, 118), (128, 127, 141), (100, 100, 112)],
    'Background': [(200, 198, 201), (193, 189, 190), (174, 172, 177), (164, 162, 165),
                   (173, 178, 181), (165, 164, 169)],
}

LABELS_Ki67 = list(PROTOTYPES_Ki67_RGB.keys())


class Patch:
    def __init__(self, image, mask, bounding_box, expert_label=None):
        self.image = image
        self.mask = mask
        self.bounding_box = bounding_box
        self.expert_label = expert_label

    def cropped_image(self):
        return crop(self.image, self.bounding_box)

    def cropped_mask(self):
        return crop(self.mask, self.bounding_box)

    def save(self, folder):
        os.makedirs(os.path.join(folder, 'image'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'data'), exist_ok=True)
        # Save image
        image_hash = hash_np(self.image)
        image_path = os.path.join(folder, 'image', f'{image_hash}.png')
        if not os.path.exists(image_path):
            imsave(image_path, arr=self.image)
        # Save mask
        mask_hash = hash_np(self.mask)
        mask_path = os.path.join(folder, 'mask', f'{mask_hash}.png')
        if not os.path.exists(mask_path):
            imsave(mask_path, arr=self.mask)
        # Save data
        data = {
            'image_hash': image_hash,
            'mask_hash': mask_hash,
            'bounding_box': self.bounding_box,
            'expert_label': self.expert_label,
        }
        data_folder = os.path.join(folder, 'data')
        num_patches = len(os.listdir(data_folder)) - 2
        suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        data_path = os.path.join(data_folder, f'{num_patches:03d}-{suffix}.json')
        with open(data_path, 'w') as outfile:
            json.dump(data, outfile)

    @staticmethod
    def load(folder, data_as_dict):
        # TODO: optimize to load image/mask only once: use load_all to load them, and then just select one here.
        image = imread(os.path.join(folder, 'image', f"{data_as_dict['image_hash']}.png"))
        mask = imread(os.path.join(folder, 'mask', f"{data_as_dict['mask_hash']}.png"))
        return Patch(
            image=image,
            mask=mask,
            bounding_box=data_as_dict['bounding_box'],
            expert_label=data_as_dict['expert_label'],
        )

    @staticmethod
    def load_all(folder):
        data_folder = os.path.join(folder, 'data')
        patches = []
        for data_fname in os.listdir(data_folder):
            with open(os.path.join(data_folder, data_fname), 'r') as outfile:
                data_as_dict = json.load(outfile)
                patches.append(Patch.load(folder=folder, data_as_dict=data_as_dict))
        return patches


class PatchClassifier:
    def __init__(self, patch_size=(101, 101)):
        self._patch_size = patch_size
        self.labelled_patches = []     # List of Patches

    def label_patches(self, patches, max_patches=None):
        if max_patches is not None:
            patches = itertools.islice(patches, max_patches)
        for patch in patches:
            assert isinstance(patch, Patch)
            os.makedirs(root_dir(FOLDER_TEMP_DATA), exist_ok=True)
            path_patch = root_dir(FOLDER_TEMP_DATA, 'patch.png')
            path_outln = root_dir(FOLDER_TEMP_DATA, 'outln.png')
            images = {
                path_patch: patch.cropped_image(),
                path_outln: outline_regions(image=patch.cropped_image(),
                                            region_labels=patch.cropped_mask())
            }
            [imsave(fname=rel_path, arr=resize(img, output_shape=(600, 600))) for rel_path, img in images.items()]
            labels = LABELS_Ki67 + ['Not well defined']
            while patch.expert_label not in labels:
                # noinspection PyTypeChecker
                gui = buttonbox("Ki67 - Manual labelling", choices=labels,
                                images=list(images.keys()), run=False)
                gui.ui.set_pos("+0+0")
                patch.expert_label = gui.run()
            print(patch.expert_label)
            self.labelled_patches.append(patch)

    def classify(self, patch):
        raise NotImplementedError('Override this method!')

    def save_labelled_patches(self, training_label):
        data_dir = root_dir(FOLDER_LABELLED_DATA, training_label)
        os.makedirs(data_dir, exist_ok=False)
        for patch in self.labelled_patches:
            patch.save(folder=data_dir)

    def load_labelled_patches(self, training_label):
        data_dir = root_dir(FOLDER_LABELLED_DATA, training_label)
        self.labelled_patches = Patch.load_all(folder=data_dir)

    def patches(self, image, region_labels):
        # Generator yielding (r_left, r_right, c_left, c_right) given a mask of labels
        for label in range(np.max(region_labels)):
            region = region_labels == label
            # Bounding box of region
            rows = np.any(region, axis=1)
            cols = np.any(region, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            r_left = (rmin + rmax - self._patch_size[0])//2
            r_right = r_left + self._patch_size[0]
            # Padded bounding box
            c_left = (cmin + cmax - self._patch_size[1])//2
            c_right = c_left + self._patch_size[1]
            # Corrected (ie. within image) padded bounding box
            if r_left < 0:
                r_right -= r_left
                r_left -= r_left
            if c_left < 0:
                c_right -= c_left
                c_left -= c_left
            if r_right >= region_labels.shape[0]:
                r_left -= (r_right - region_labels.shape[0] + 1)
                r_right -= (r_right - region_labels.shape[0] + 1)
            if c_right >= region_labels.shape[1]:
                c_left -= (c_right - region_labels.shape[1] + 1)
                c_right -= (c_right - region_labels.shape[1] + 1)
            bounding_box = r_left.item(), r_right.item(), c_left.item(), c_right.item()
            yield Patch(
                image=image,
                mask=region_labels == label,
                bounding_box=bounding_box)


class ClosestCentroidClassifier():
    # TODO based on pixelwise_closest_centroid
    pass

def pixelwise_closest_centroid(image, centroids_rgb):
    # Centroids must be a dictionary { 'name of group': [(corresponding, rgb, triplets), ...] }
    # TODO: do this in the CIELAB color space
    distance = np.full(image.shape[:2] + (len(centroids_rgb),), fill_value=np.inf, dtype='float')
    image = img_as_float(image)
    for idx_c, list_reference_color_rgb in enumerate(centroids_rgb.values()):
        for reference_color_rgb in list_reference_color_rgb:
            reference_color_rgb = img_as_float(np.asarray([c/255 for c in reference_color_rgb]).reshape((1, 1, 3)))
            distance[:, :, idx_c] = np.minimum(
                distance[:, :, idx_c],
                np.sum(np.square(image - reference_color_rgb), axis=2))
    return np.argmin(distance, axis=2), distance

