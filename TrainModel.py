import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
# from keras.applications.vgg16 import VGG16
#https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5
from keras.layers import Input, Flatten, Dense, Dropout, AveragePooling2D, Average, GlobalAveragePooling2D
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
from keras import metrics
from sklearn.metrics import confusion_matrix
import scipy.misc
from keras.utils import multi_gpu_model

def top_5_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

def top_3_acc(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

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

def read_pickle(pickle_path, pickle_file):

    print("Reading %s" % (pickle_path + '/' + pickle_file))

    with open(os.path.join(pickle_path, pickle_file), 'rb') as p:
        pickled_data = pickle.load(p)
    train_data = pickled_data[0]
    train_label = pickled_data[1]
    train_label = np.array(train_label)
    test_data = pickled_data[2]
    test_label = pickled_data[3]
    test_label = np.array(test_label)
    return train_data, train_label, test_data, test_label

def group_imresize(data, resolution):
    print('Resizing Data')
    data = np.array(data)
    tmp_data = []
    for i in range(np.shape(data)[0]):
        if i % 10000 == 0:
            print('image {} is resized'.format(i))
        x_img = scipy.misc.imresize(data[i], [resolution, resolution])
        tmp_data.append(x_img)
    return np.array(tmp_data)

def main():

    os_config()

    if os.path.exists('log') == False:
        os.mkdir('log')
    if os.path.exists('acc_class') == False:
        os.mkdir('acc_class')

    batch_size = 32
    epochs = 200
    target_class_num = 2000

    model_select = 'Xception' #InceptionV3, Xception, InceptionResNetV2, NASNetLarge VGG16
    LogFileName = model_select + '_e' + str(epochs) + 'c' + str(target_class_num)
    model_name = model_select + '_e' + str(epochs) + 'c' + str(target_class_num) + '.h5'

    train_data, train_label, test_data, test_label = read_pickle("/HDD/joshua/DemoTop" + str(target_class_num), "data.pickle")

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    # train_data, train_label, organ_label, test_data, test_label, organ_test_label = read_pickle("OrganTop" + str(target_class_num), "data.pickle")

    train_label = one_hot_encode(train_label, target_class_num)
    test_label = one_hot_encode(test_label, target_class_num)

    print("training data shape: {}".format(np.shape(train_data)))
    print("training label shape: {}".format(np.shape(train_label)))
    print("test data shape: {}".format(np.shape(test_data)))
    print("test label shape: {}".format(np.shape(test_label)))

    # normalize inputs from 0-255 to 0.0-1.0
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    input = Input(shape=(227,227,3), name='image_input')

    if model_select == 'VGG16':
        from keras.applications.vgg16 import VGG16
        transfer_model = VGG16(weights='imagenet', include_top=False)
        transfer_model.summary()
        for layer in transfer_model.layers:
            layer.trainable = True
        output_transfer = transfer_model(input)
        output_transfer = Flatten(name='flatten')(output_transfer)
        output_transfer = Dense(4096, name='my_dense1')(output_transfer)
        output_transfer = Dense(4096, name='my_dense2')(output_transfer)

    if model_select == 'NASNetLarge':
    # model
        from keras.applications.nasnet import NASNetLarge
        input = Input(shape=(331,331,3), name='image_input')
        train_data = group_imresize(train_data, 331)
        test_data = group_imresize(test_data, 331)

        print("New training data shape: {}".format(np.shape(train_data)))
        print("New test data shape: {}".format(np.shape(test_data)))
        
        transfer_model = NASNetLarge(weights='imagenet', include_top=False)
        # transfer_model.summary()
        for layer in transfer_model.layers:
            layer.trainable = True

        output_transfer = transfer_model(input)
        output_transfer = GlobalAveragePooling2D(name='my_GlobalAveragePooling')(output_transfer)

    if model_select == 'InceptionV3':
    # model
        from keras.applications.inception_v3 import InceptionV3
        transfer_model = InceptionV3(weights='imagenet', include_top=False)
        # transfer_model = InceptionV3(weights=None, include_top=False)
        transfer_model.summary()
        for layer in transfer_model.layers:
            layer.trainable = True

        output_transfer = transfer_model(input)
        output_transfer = AveragePooling2D(pool_size=(5, 5), strides=1, name='my_averagePooling')(output_transfer)
        output_transfer = Dropout(0.4, name='my_dropOut')(output_transfer)
        output_transfer = Flatten(name='flatten')(output_transfer)
    
    if model_select == 'Xception':

        from keras.applications.xception import Xception
        transfer_model = Xception(weights='imagenet', include_top=False)
        for layer in transfer_model.layers:
            layer.trainable = True
        output_transfer = transfer_model(input)
        output_transfer = GlobalAveragePooling2D(name='my_global_pool')(output_transfer)

    if model_select == 'InceptionResNetV2':

        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        transfer_model = InceptionResNetV2(weights='imagenet', include_top=False)
        # transfer_model = InceptionResNetV2(weights=None, include_top=False)
        transfer_model.summary()
        for layer in transfer_model.layers:
            layer.trainable = True

        output_transfer = transfer_model(input)
        output_transfer = GlobalAveragePooling2D(name='my_globalAveragePooling')(output_transfer)
        # output_transfer = AveragePooling2D(pool_size=(5, 5), strides=1)(output_transfer)
        output_transfer = Dropout(0.2, name='my_dropOut')(output_transfer)

        # output_transfer = Flatten(name='flatten')(output_transfer)


    out = Dense(target_class_num, activation='softmax', name='predictions')(output_transfer)
    # Create your own model
    my_model = Model(input=input, output=out)
    my_model.summary()

    sgd = SGD(lr=0.045, decay=1e-6, momentum=0.9, nesterov=True)
    # my_model = multi_gpu_model(my_model, gpus=2)
    my_model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_3_acc, top_5_acc])

    train_datagen = ImageDataGenerator(
        # width_shift_range = 0.10,
        # height_shift_range = 0.10,
        rotation_range = 20,
        # shear_range = 0.10,
        zoom_range = 0.10,
        horizontal_flip = True,
        fill_mode = 'nearest'
        )

    val_datagen = ImageDataGenerator()

    train_datagen.fit(train_data)
    train_generator = train_datagen.flow(train_data, train_label, batch_size = batch_size)

    val_datagen.fit(test_data)
    val_generator = val_datagen.flow(test_data, test_label, batch_size = batch_size)

    checkpoint = ModelCheckpoint('/HDD/joshua/models/' + 'models_' + model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger('log/' + LogFileName + '.log')

    # steps_per_epoch should be (number of training images total / batch_size) 
    # validation_steps should be (number of validation images total / batch_size)
    
    my_model.fit_generator(train_generator,
                        steps_per_epoch= np.shape(train_data)[0] / batch_size,
                        validation_data= val_generator,
                        validation_steps= np.shape(test_data)[0] / batch_size, 
                        epochs=epochs,
                        callbacks = [csv_logger, checkpoint]
                        )
    # my_model.fit_generator(train_generator,
    #                     steps_per_epoch= np.shape(train_data)[0] / batch_size,
    #                     validation_data= val_generator,
    #                     validation_steps= np.shape(test_data)[0] / batch_size, 
    #                     epochs=epochs,
    #                     callbacks = [csv_logger, checkpoint]
    #                     )
    
    # my_model.fit(train_data, train_label,
    #           nb_epoch=40,
    #           batch_size=64,
    #           validation_data=(test_data, test_label),
    #           callbacks = [csv_logger]
    #           )

    
    test_pred = my_model.predict(test_data, batch_size = batch_size)
    # test_pred = my_model.predict(test_data, batch_size = batch_size)
    cmat = confusion_matrix(np.argmax(test_label, axis = 1), np.argmax(test_pred, axis = 1))
    acc_per_class = cmat.diagonal() / cmat.sum(axis=1)

    print(acc_per_class)
    with open('acc_class/' + LogFileName + '.txt', 'w') as f:
        for item in acc_per_class:
            f.write('{}, '.format(item))

    # my_model.save('models/' + model_name)
    # my_model = load_model('/HDD/joshua/models/' + model_name)

if __name__ == '__main__':
    main()
