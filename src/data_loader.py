import os

from skimage.io import imread

FOLDER_DATA = 'Data'
FOLDER_RESULTS = 'Results'
FOLDER_EXPERIMENTS = os.path.join(FOLDER_RESULTS, 'Experiments')
FOLDER_TEMP_DATA = os.path.join(FOLDER_RESULTS, 'Results/0000 TEMP')
FOLDER_LABELLED_DATA = os.path.join(FOLDER_RESULTS, 'Results/Labelled')

def root_dir(*args):
    return os.path.abspath(os.path.join(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, *args)))

def patient_names():
    path_data = root_dir(FOLDER_DATA)
    return [name for name in os.listdir(path_data)
            if os.path.isdir(os.path.join(path_data, name))]

def image_paths(patient_name):
    path_images = root_dir(FOLDER_DATA, patient_name)
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
