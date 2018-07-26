from __future__ import print_function
from __future__ import absolute_import
import csv
import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


import warnings

from keras.models import Model, Sequential, load_model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation, Conv2D, SeparableConv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
# from scipy.misc import imsave

import keras
from keras.layers import Flatten, Dropout, AveragePooling2D, Average
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
import numpy as np
import pickle
from keras import metrics
from sklearn.metrics import confusion_matrix
import scipy.misc
from keras.callbacks import ModelCheckpoint


def top_3_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_5_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

def os_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

xception_model = load_model('/HDD/joshua/models/models_Xception_e200c2000.h5', custom_objects={'top_3_acc': top_3_acc, 'top_5_acc': top_5_acc})

layer_name = 'my_global_pool'
intermediate_layer_model = Model(inputs=xception_model.input,
                                 outputs=xception_model.get_layer(layer_name).output)


dataNum = np.zeros((2000,1))

for i in range(2000):
    SimilarData = []
    print('Feature extracting class {} ...'.format(i))
    for dirPath, dirNames, fileNames in os.walk('/HDD/joshua/DemoPlantsImages/' + str(i)):
        tmpfileNames = []

        for jpg in fileNames:
            img_path = os.path.join(dirPath, jpg)
            if jpg.endswith(".jpg"):
                tmpfileNames.append('DemoPlantsImages/' + str(i) + '/' + jpg)
                # print(tmpfileNames[int(dataNum[i])])
                test_img = scipy.misc.imread(img_path, mode='RGB')
                test_img = scipy.misc.imresize(test_img, [227, 227])
                test_img = np.array(test_img) / 255.0
                SimilarData.append(test_img)
                dataNum[i] += 1

    SimilarData = np.array(SimilarData)
    features = intermediate_layer_model.predict(SimilarData, batch_size = 32)

    write_pickle = (tmpfileNames, features)
    print(np.shape(features))

    with open(os.path.join('/HDD/joshua/DemoPlantsImages/Features', str(i) + '.pickle'), 'wb') as p:
            pickle.dump(write_pickle, p, protocol = 4)
        












