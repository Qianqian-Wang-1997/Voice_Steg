import sys
import csv
import datetime
import random
import pickle
import os
from test_metric import *
import sklearn as sk
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from keras.callbacks import Callback
import numpy
from keras.callbacks import ModelCheckpoint
from addattention import *
import tensorflow as tf
from keras import regularizers
import pdb
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPool1D
from keras.layers.convolutional import Conv1D
from keras.layers import Embedding, Dense, Flatten, Input, PReLU
from sklearn.svm import SVC
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.layers.merge import concatenate
from keras.layers import Reshape, Dense, Dropout, LSTM, Bidirectional, AveragePooling1D, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Activation
import numpy as np
# np.set_printoptions(threshold=np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from cnnmodel import TextCNN:
# from keras_utils import Capsule, AttentionWithContext,Attention


Embedding_rate = 100000
SAMPLE_LENGTH = 1000   # The sample length (ms)
BATCH_SIZE = 256         # batch size
ITER = 50                # number of iteration
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


File_Embed = "./cat/train/Emb"
File_NoEmbed = "./cat/train/NoEmb"

FOLDERS = [
    # The folder that contains positive data files.
    {"class": 1, "folder": File_Embed},
    # The folder that contains negative data files.
    {"class": 0, "folder": File_NoEmbed}
]

testFile = "./cat/test2"

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


def build_model(NUM_FILTERS):
    model_input = Input(shape=(int(SAMPLE_LENGTH/10), 5))  # PD = 2, LCP = 3

    # x= Bidirectional(LSTM(hi, return_sequences=True), merge_mode='concat')(model_input)
    rnn_feature = Bidirectional(
        LSTM(hidden_num, return_sequences=True))(model_input)
    pooled_outputs = []
    emb_x = Embedding(256, 64)(model_input)
    x = Reshape((int(SAMPLE_LENGTH/10), -1))(emb_x)
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
    x = Dense(NUM_FILTERS)(x)
    x = BatchNormalization()(x)
    feats = Activation('relu')(x)

    source_classifier = Dense(
        NUM_FILTERS, activation='linear', name='mo4')(feats)
    source_classifier = BatchNormalization(name='mo5')(source_classifier)
    source_classifier = Activation('relu', name='mo6')(source_classifier)
    source_classifier = Dropout(0.5)(source_classifier)
    source_classifier = Dense(
        num_class, activation='softmax', name='mo')(source_classifier)

    domain_classifier = Dense(
        NUM_FILTERS, activation='linear', name='do4')(feats)
    domain_classifier = BatchNormalization(name='do5')(domain_classifier)
    domain_classifier = Activation('relu', name='do6')(domain_classifier)
    domain_classifier = Dropout(0.5)(domain_classifier)
    domain_classifier = Dense(
        num_class, activation='softmax', name='do')(domain_classifier)

    comb_model = Model(model_input, [source_classifier, domain_classifier])
    comb_model.summary()
    comb_model.compile(optimizer='adam', loss={'mo': 'binary_crossentropy', 'do': 'binary_crossentropy'},
                       loss_weights={'mo': 1, 'do': 1}, metrics=['accuracy'])

    source_model = Model(model_input, [source_classifier])
    source_model.compile(loss={'mo': 'binary_crossentropy'},
                         optimizer='adam', metrics=['accuracy'])

    domain_model = Model(model_input, [domain_classifier])
    domain_model.compile(optimizer="Adam", loss={
                         'do': 'binary_crossentropy'}, metrics=['accuracy'])

    embeddings_model = Model(model_input, [x])
    embeddings_model.compile(
        optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])

    return comb_model, source_model, domain_model, embeddings_model


def batch_generator(data, batch_size):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns batches of the same size
    This
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(
            all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


'''
Pruned RNN-SM training and testing
'''

if __name__ == '__main__':

    print("Embedding_rate,SAMPLE_LENGTH,flag_str:",
          Embedding_rate, SAMPLE_LENGTH, flag_str)
    pklfilex = './data/xtrain_new_cat.pkl'
    pklfiley = './data/ytrain_new_cat.pkl'
    pklfile_test_x = './data/test2_cat.pkl'

# train
    if not os.path.exists(pklfilex):
        print("processing pkl")
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
        print("Got it")
        # all_files = [(item, folder["class"])]
        np_all_samples_x = pickle.load(open(pklfilex, 'rb'))
        np_all_samples_y = pickle.load(open(pklfiley, 'rb'))
        np_all_test_samples_x = pickle.load(open(pklfile_test_x, 'rb'))

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

    x_val = np_all_samples_x[0: sub_file_num]  # The samples for testing
    # The label of the samples for testing
    y_val_ori = np_all_samples_y[0: sub_file_num]

    x_train = np_all_samples_x[sub_file_num:]  # The samples for training
    # The label of the samples for training
    y_train_ori = np_all_samples_y[sub_file_num:]

    x_test = np_all_test_samples_x
  #  y_test_ori = np_all_test_samples_y

    print("x_train.shape:", x_train.shape)
    print("y_train.shape", y_train_ori.shape)
    print("x_val.shape", x_val.shape)
    print("y_val_ori", y_val_ori.shape)
    print("y_val:", y_val_ori.sum())
    print("y_train_ori:", y_train_ori.sum())
    print("x_test.shape:", x_test.shape)

    y_train = np_utils.to_categorical(y_train_ori, num_classes=2)
    y_val = np_utils.to_categorical(y_val_ori, num_classes=2)
  #  y_test = np_utils.to_categorical(y_test_ori, num_classes=2)

    print("Building model")
    model, source_model, domain_model, embedding_model = build_model(
        NUM_FILTERS)

   # y_class_dummy = np.ones((len(x_test), 5))
    y_adversarial_1 = np_utils.to_categorical(
        np.array(([1]*BATCH_SIZE + [0]*BATCH_SIZE)))
    print(y_adversarial_1)
    #######
    sample_weights_class = np.array(([1] * BATCH_SIZE + [0] * BATCH_SIZE))
    sample_weights_adversarial = np.ones((BATCH_SIZE * 2,))

    print("s_batching")
    S_batches = batch_generator(
        [x_train, y_train], BATCH_SIZE)
    ######
    print('t_batching')
    T_batches = batch_generator(
        [x_test, np.zeros(shape=(len(x_test), 2))], BATCH_SIZE)

    enable_dann = True

    for i in range(ITER):
        # # print(y_class_dummy.shape, ys.shape)
        print(i, ' iter')
        ####################################################
        for j in range(560):
           # print(j)

            y_adversarial_2 = np_utils.to_categorical(
                np.array(([0] * BATCH_SIZE + [1] * BATCH_SIZE)))

            X0, y0 = next(S_batches)
            X1, y1 = next(T_batches)

            X_adv = np.concatenate([X0, X1])
            y_class = np.concatenate([y0, np.zeros_like(y0)])

            adv_weights = []
            for layer in model.layers:
                if (layer.name.startswith("do")):
                    adv_weights.append(layer.get_weights())

            if(enable_dann):
                # note - even though we save and append weights, the batchnorms moving means and variances
                # are not saved throught this mechanism
                stats = model.train_on_batch(X_adv, [y_class, y_adversarial_1],
                                             sample_weight=[sample_weights_class, sample_weights_adversarial])
                k = 0
                for layer in model.layers:
                    if (layer.name.startswith("do")):
                        layer.set_weights(adv_weights[k])
                        k += 1

                class_weights = []

                for layer in model.layers:
                    if (not layer.name.startswith("do")):
                        class_weights.append(layer.get_weights())

                stats2 = domain_model.train_on_batch(X_adv, [y_adversarial_2])

                k = 0
                for layer in model.layers:
                    if (not layer.name.startswith("do")):
                        layer.set_weights(class_weights[k])
                        k += 1

            else:
                source_model.train_on_batch(X0, y0)

        print('predicting target')
        y_test_hat_t = source_model.predict(x_test)
        print(y_test_hat_t)
        y_test_hat_t = np.argmax(y_test_hat_t, axis=1)
        print(y_test_hat_t)
       # pdb.set_trace()
        print('saving txt')
        np.savetxt('./result_1010/results_' +
                   str(i)+'.txt', y_test_hat_t, '%d')

        print('predicting source')
        y_test_hat_s = source_model.predict(x_val)
        y_test_hat_s = np.argmax(y_test_hat_s, axis=1)

        print("Iteration %d, source accuracy =  %.3f, target accuracy = %s" % (
            i, accuracy_score(y_val_ori, y_test_hat_s), np.sum(y_test_hat_t == 0)))
#        print("Iteration %d, target accuracy = %.3f" % (
#            i, accuracy_score(y_test_ori, y_test_hat_t)))

  #
  #  for i in range(ITER):
   #     checkpoint = ModelCheckpoint(filepath='./model/cat_best.hdf5', monitor='val_acc',
    #                                 verbose=1, save_best_only=True, mode='auto', period=1)
    #   callbacks_list = [checkpoint]
#

 #       comb_model.fit(x_train, y_train, batch_size=BATCH_SIZE,
  #                     nb_epoch=1, validation_data=(x_test, y_test), callbacks=callbacks_list)
   #     ys_pred = comb_model.predict(x_test)
    #    ys_pred = np.argmax(ys_pred, axis=1)
    #   print(np.sum(y_pred == 0))
    #  print_result(y_test_ori, y_pred)
