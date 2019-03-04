import os

import skimage

RELATIVE_DATA_PATH = 'Data'

def patient_names():
    path_data = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, RELATIVE_DATA_PATH))
    return [name for name in os.listdir(path_data)
            if os.path.isdir(os.path.join(path_data, name))]

def image_paths(patient_name):
    path_images = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, RELATIVE_DATA_PATH, patient_name))
    return [os.path.join(path_images, name) for name in os.listdir(path_images)
            if os.path.splitext(os.path.join(path_images, name))[1] == '.jpg']

def num_images(patient_name):
    return len(image_paths(patient_name))

def images(patient_name, idcs=None):
    path_images = image_paths(patient_name)
    if idcs is not None:
        path_images = path_images[idcs]
    for path in path_images:
        yield path, skimage.io.imread(path)
