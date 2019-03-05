import os

import skimage
from skimage import img_as_float
from skimage.io import imread

RELATIVE_DATA_PATH = 'Data'

def patient_names():
    path_data = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, RELATIVE_DATA_PATH))
    return [name for name in os.listdir(path_data)
            if os.path.isdir(os.path.join(path_data, name))]

def image_paths(patient_name):
    path_images = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, RELATIVE_DATA_PATH, patient_name))
    return [os.path.join(path_images, name) for name in os.listdir(path_images)
            if os.path.splitext(os.path.join(path_images, name))[1] == '.jpg']

def num_images(patient_name):
    return len(image_paths(patient_name))

def images(patient_name, max_images=None):
    path_images = image_paths(patient_name)
    if max_images is not None:
        path_images = path_images[0:max_images]
    for path in path_images:
        yield path, imread(path)
