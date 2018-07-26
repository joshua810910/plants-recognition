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


def top_2_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_5_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

def os_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

def one_hot_encode(x_label, classNumber):
    train_label = []
    for i in range(np.shape(x_label)[0]):
        train_tmp_label = np.zeros(classNumber)
        train_tmp_label[x_label[i]] = 1
        train_label.append(train_tmp_label)
    train_label = np.array(train_label)

    return train_label

img = []
x_img = scipy.misc.imread(sys.argv[1], mode='RGB')


# x_img = np.array(x_img)
# if np.shape(x_img)[0] > np.shape(x_img)[1]:



x_img = scipy.misc.imresize(x_img, [227, 227])
x_img = np.array(x_img)
print(np.shape(x_img))
img.append(x_img)
img = np.array(img) / 255.0
print(np.shape(img))

xception_model = load_model('models/models_Xception_e200c2000.h5', custom_objects={'top_3_acc': top_3_acc, 'top_5_acc': top_5_acc})
prediction = xception_model.predict(img, batch_size = 32)
top5acc = np.argsort(prediction)
top5acc = np.flip(top5acc[0], axis=0)
top5acc = top5acc[0:5]
print(np.argmax(prediction[0]))
print(top5acc)

RowData = []
with open('dataIndex_top.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        row = ''.join(row[1])
        # print(row.encode(sys.stdin.encoding, "replace").decode(sys.stdin.encoding))
        RowData.append(row)
        # print(row)
for i in range(5):
    print(RowData[top5acc[i]].encode(sys.stdin.encoding, "replace").decode(sys.stdin.encoding))




layer_name = 'my_global_pool'
intermediate_layer_model = Model(inputs=xception_model.input,
                                 outputs=xception_model.get_layer(layer_name).output)
target_features = intermediate_layer_model.predict(img)


# dataNum = np.zeros((5,1))

for i in range(5):
    with open('Features/'+str(top5acc[i]) + '.pickle', 'rb') as p:
        pickled_data = pickle.load(p)
    tmpfileNames = pickled_data[0]
    features = pickled_data[1]

    # SimilarData = []
    # for dirPath, dirNames, fileNames in os.walk('DemoPlantsImages/' + str(top5acc[i])):
    #     tmpfileNames = []
    #     for jpg in fileNames:
    #         img_path = os.path.join(dirPath, jpg)
    #         if jpg.endswith(".jpg"):
    #             tmpfileNames.append(img_path)
    #             test_img = scipy.misc.imread(img_path, mode='RGB')
    #             test_img = scipy.misc.imresize(test_img, [227, 227])
    #             test_img = np.array(test_img) / 255.0
    #             SimilarData.append(test_img)
    #             dataNum[i] += 1

    # SimilarData = np.array(SimilarData)
    # features = intermediate_layer_model.predict(SimilarData, batch_size = 32)

    mindistance = float("inf")
    # for j in range(int(dataNum[i])):
    for j in range(np.shape(features)[0]):
        distance = np.linalg.norm(target_features[0] - features[j])
        if distance < mindistance:
            mindistance = distance
            min_index = j
    print(tmpfileNames[min_index])



# SimilarData.append(x_img / 255.0)
# SimilarData = np.array(SimilarData)




# Data0 = features[0:dataNum[0]]
# Data1 = features[dataNum[0]:dataNum[1]]
# Data2 = features[dataNum[1]:dataNum[2]]
# Data3 = features[dataNum[2]:dataNum[3]]
# Data4 = features[dataNum[3]:dataNum[4]]
# Target = features[dataNum[4]:dataNum[4]]



# mindistance = float("inf")
# for i in range(int(dataNum[1])):
#     distance = np.linalg.norm(Target - Data1[i])
#     if distance < mindistance:
#         mindistance = distance
#         min_index1 = i

# mindistance = float("inf")
# for i in range(int(dataNum[1])):
#     distance = np.linalg.norm(Target - Data1[i])
#     if distance < mindistance:
#         mindistance = distance
#         min_index1 = i

# print(dataNum[0])
# print(dataNum[1])
# print(dataNum[2])
# print(dataNum[3])
# print(dataNum[4])
# print(np.shape(SimilarData))



############################################################################################
# layer_name = 'my_global_pool'
# intermediate_layer_model = Model(inputs=xception_model.input,
#                                  outputs=xception_model.get_layer(layer_name).output)
# # test_output = intermediate_layer_model.predict(test_data[10513:10514])
# test_output = intermediate_layer_model.predict(img, batch_size = 32)
# print('finish mid prediction')
############################################################################################





# mindistance = float("inf")
# test_target = test_output[10513]

# for j in range(np.shape(train_data)[0]):
#     if np.argmax(train_label[j]) == 423:
#         # layer_name = 'my_global_pool'
#         # intermediate_layer_model = Model(inputs=my_model.input,
#         #                                  outputs=my_model.get_layer(layer_name).output)
#         # intermediate_output = intermediate_layer_model.predict(train_data[i:i+1])

#         distance = np.linalg.norm(test_target - train_output[j])
#         if distance < mindistance:
#             mindistance = distance
#             min_index = j
#         print('{},{} {}'.format(j, distance, mindistance))

# imsave('/HDD/joshua/closed_set/T' + str(int(np.argmax(train_label[min_index]))) + '_test.jpg', test_data[10513]*255)
# imsave('/HDD/joshua/closed_set/T' + str(int(np.argmax(train_label[min_index]))) + '_train.jpg', train_data[min_index]*255)

# train_data_append = []
# train_output_append = []
# print('appending data')
# for i in range(500):
# 	train_data_append.append([])
# 	train_output_append.append([])

# for i in range(np.shape(train_data)[0]):
# 	train_data_append[np.argmax(train_label[i])].append(train_data[i])
# 	train_output_append[np.argmax(train_label[i])].append(train_output[i])

# train_data_append = np.array(train_data_append)
# train_output_append = np.array(train_output_append)

# print('start scanning')

# for i in range(np.shape(test_data)[0]):
# 	if np.equal(np.argmax(prediction[i]), np.argmax(test_label[i])) == False:
# 		print('image {} predict wrong, saved to {}'.format(i, int(np.argmax(prediction[i]))))

# 		mindistance = float("inf")
		
# 		for j in range(np.shape(train_data_append[int(np.argmax(prediction[i]))])[0]):
# 			distance = np.linalg.norm(test_output[i] - train_output_append[np.argmax(prediction[i])][j])
# 			if distance < mindistance:
# 				mindistance = distance
# 				min_index = j

# 		imageToSaved = np.zeros((227, 227*2+3, 3))
# 		for w in range(227):
# 			for l in range(227):
# 				for d in range(3):
# 					imageToSaved[w][l][d] = test_data[i][w][l][d] * 255
# 					imageToSaved[w][l+227+3][d] = train_data_append[np.argmax(prediction[i])][min_index][w][l][d] * 255

# 		print('{},{} {}'.format(j, distance, mindistance))
# 		total_err += 1
# 		error_rate[np.argmax(prediction[i])] += 1
# 		# imsave('/workspace/Error/' + str(int(np.argmax(prediction[i]))) + '/' + str(i) + '.jpg', imageToSaved)
# 		imsave('/workspace/Error2/' + str(i) + '.jpg', imageToSaved)

# with open('/workspace/Error2/each.txt', 'w') as f:
#     for item in error_rate:
#         f.write('{}\n '.format(item))

# print(total_err)
# print((np.shape(test_data)[0] - total_err) / np.shape(test_data)[0])


