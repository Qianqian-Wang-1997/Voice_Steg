import pickle
import numpy as np
import h5py
from keras.models import load_model
import tensorflow as tf
import os
import random
import sys

File_test = "./test"
FOLDERS = [{"class": "un": File_test}]


def read_model():
    model = load_model('/modelbest.h5')
    return model


def get_file_list(folder):
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list


def parse_sample(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    sample = []
    for line in lines:
        line_split = line.strip("\r\n\t").strip().split(" ")
       # line_split = line_splitx[:5]
        sample.append(line_split)
    return sample


'''
Save variable in pickle
'''


def save_variable(file_name, variable):
    file_object = open(file_name, "wb")
    pickle.dump(variable, file_object)
    file_object.close()


if __name__ == '__main__':
    test_files = [(item, folder)
                  for folder in FOLDERS for item in get_file_list(folder["folder"])]
    random.shuffle(test_files)
    save_variable('test_files_test.pkl', test_files)

    all_samples = [(parse_sample(item))]
