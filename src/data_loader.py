import csv
import os
import re

from skimage.io import imread

FOLDER_DATA = 'Data'
FOLDER_RESULTS = 'Results'


def FOLDER_EXPERIMENTS(version):
    return os.path.join(FOLDER_RESULTS, f'Experiments v{version}')

FOLDER_TEMP_DATA = os.path.join(FOLDER_RESULTS, '0000 TEMP')
FOLDER_LABELLED_DATA = os.path.join(FOLDER_RESULTS, 'Labelled')


# Generic
###


def root_dir(*args):
    return os.path.abspath(os.path.join(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, *args)))


def sample_names():
    path_data = root_dir(FOLDER_DATA)
    return [name for name in os.listdir(path_data)
            if os.path.isdir(os.path.join(path_data, name))]


# Image retrieving
###


def image_paths(sample_name):
    path_images = root_dir(FOLDER_DATA, sample_name)
    return [os.path.join(path_images, name) for name in os.listdir(path_images)
            if os.path.splitext(os.path.join(path_images, name))[1] == '.jpg']


def num_images(sample_name):
    return len(image_paths(sample_name))


def images(patient_name, max_images=None):
    path_images = image_paths(patient_name)
    if max_images is not None:
        path_images = path_images[0:max_images]
    for path in path_images:
        yield path, imread(path)


# Data retrieving
###

def read_from_csv(NCH, csv_row):
    csv_path = root_dir(FOLDER_DATA, 'GLIOMES PET - ANONIMITZAT.csv')
    with open(csv_path, encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=";")
        for row in csv_reader:
            if row['NCH'] == NCH:
                return row[csv_row]
    raise AttributeError()

def get_expert_Ki67(sample_name):
    m = re.match(r'(?P<NCH>[0-9]*) ?-(?P<Zone>[AB])[0-9]', sample_name)
    return read_from_csv(NCH=m.group('NCH'), csv_row=f"Zona {m.group('Zone')} - Ki67")
