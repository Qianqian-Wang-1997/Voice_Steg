
import numpy as np
import os
import pickle
import random
import datetime
import os
import random
import pickle
import csv
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from cnnmodel import TextCNN
from keras.layers import Reshape, Dense, Dropout, LSTM, Bidirectional, AveragePooling1D, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.svm import SVC
from keras.layers import Embedding, Dense, Flatten, Input
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
# from keras_utils import Capsule, AttentionWithContext,Attention
import pdb
from keras import regularizers
import tensorflow as tf
from addattention import *
from keras.callbacks import ModelCheckpoint

import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import sklearn as sk

from test_metric import *

Embedding_rate = 100000
SAMPLE_LENGTH = 1000   # The sample length (ms)
BATCH_SIZE = 256         # batch size
ITER = 20                # number of iteration
FOLD = 5                # = NUM_SAMPLE / number of testing samples
NUM_FILTERS = 32
FILTER_SIZES = [5, 4, 3]
num_class = 2
vocab_size = 256
hidden_num = 64
k = 2
tag = 'lstm+cnn_iter'
flag_str = 'ch'

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


File_Embed = "./train_PD/Emb"
File_NoEmbed = "./train_PD/NoEmb"

FOLDERS = [
    # The folder that contains positive data files.
    {"class": 1, "folder": File_Embed},
    # The folder that contains negative data files.
    {"class": 0, "folder": File_NoEmbed}
]


'''
Get the paths of all files in the folder
'''


def get_file_list(folder):
    file_list = []
    for file in os.listdir(folder):
        file_list.append(os.path.join(folder, file))
    return file_list


'''
Read codeword file
-------------
input
    file_path
        The path to an ASCII file.
        Each line contains three integers: x1 x2 x3, which are the three codewords of the frame.
        There are (number of frame) lines in total.
output
    the list of codewords
'''


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


'''
Pruned RNN-SM training and testing
'''
if __name__ == '__main__':

    print("Embedding_rate,SAMPLE_LENGTH,flag_str:",
          Embedding_rate, SAMPLE_LENGTH, flag_str)
    pklfilex = './data/xtrain.pkl'
    pklfiley = './data/ytrain.pkl'

    if not os.path.exists(pklfilex):
        print("我到了！")
        all_files = [(item, folder["class"])
                     for folder in FOLDERS for item in get_file_list(folder["folder"])]
        random.shuffle(all_files)
        save_variable('all_files_train.pkl', all_files)
        # pdb.set_trace()
        all_samples_x = [(parse_sample(item[0])) for item in all_files]
        print("parse")
        all_samples_y = [item[1] for item in all_files]
        np_all_samples_x = np.asarray(all_samples_x)
        np_all_samples_y = np.asarray(all_samples_y)
        save_variable(pklfilex, np_all_samples_x)
        save_variable(pklfiley, np_all_samples_y)
    else:
        print("OH FUCK")
        # all_files = [(item, folder["class"])]
        np_all_samples_x = pickle.load(open(pklfilex, 'rb'))
        np_all_samples_y = pickle.load(open(pklfiley, 'rb'))

    if len(np_all_samples_x.shape) < 2:
        index = [i for i in range(np_all_samples_x.shape[0]) if len(
            np_all_samples_x[i]) < 1]
        new_x = np.delete(np_all_samples_x, index)
        np_all_samples_x = np.asarray(list(new_x))
        new_y = np.delete(np_all_samples_y, index)
        np_all_samples_y = np.asarray(list(new_y))

    file_num = len(np_all_samples_y)
    sub_file_num = int(file_num / FOLD)
    print("NP_ALL_SAMPLES_X.SHAPE:", np_all_samples_x.shape)

    dataSize = np_all_samples_x.shape

    x_test = np_all_samples_x[0: sub_file_num]  # The samples for testing
    # The label of the samples for testing
    y_test_ori = np_all_samples_y[0: sub_file_num]

    x_train = np_all_samples_x[sub_file_num:]  # The samples for training
    # The label of the samples for training
    y_train_ori = np_all_samples_y[sub_file_num:]

    print("x_train.shape:", x_train.shape)
    print("y_train.shape", y_train_ori.shape)
    print("x_test.shape", x_test.shape)
    print("y_test_ori", y_test_ori.shape)
    print("y_test:", y_test_ori.sum())
    print("y_train_ori:", y_train_ori.sum())

    y_train = np_utils.to_categorical(y_train_ori, num_classes=2)
    y_test = np_utils.to_categorical(y_test_ori, num_classes=2)

    print("Building model")

    model_input = Input(shape=(int(SAMPLE_LENGTH/10), 2))  # PD = 2, LCP = 3

    # x= Bidirectional(LSTM(hi, return_sequences=True), merge_mode='concat')(model_input)
    rnn_feature = Bidirectional(
        LSTM(hidden_num, return_sequences=True))(model_input)
    pooled_outputs = []
    for filter_size in FILTER_SIZES:

        x = Conv1D(NUM_FILTERS*2, filter_size, padding='same')(rnn_feature)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = AveragePooling1D(int(x.shape[1]))(x)
        # x = GlobalAveragePooling1D()(x)
        x = AttLayer()(x)
        x = Reshape((1, -1))(x)
        pooled_outputs.append(x)

    merged = concatenate(pooled_outputs)
    x = Flatten()(merged)
    x = BatchNormalization()(x)
    x = Dense(NUM_FILTERS)(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    model_output = Dense(num_class, activation='softmax')(x)
    model = Model(model_input, model_output)

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=["accuracy"])

    print("Training")
    for i in range(ITER):
        checkpoint = ModelCheckpoint(filepath='testbest.hdf5', monitor='accuracy',
                                     verbose=1, save_best_only='True', mode='auto', period=1)
        callbacks_list = [checkpoint]

        model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                  nb_epoch=1, validation_data=(x_test, y_test), callbacks=callbacks_list)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        print_result(y_test_ori, y_pred)
