import os,re,sys
import scipy.misc
import random
import numpy as np
from PIL import Image
import pickle
import csv
from scipy.misc import imsave

def read_data():

    validation_rate = 0.1
    resolute = 227
    train_folder = ['171107', '171113', '171121', '171128', '171204', '171211', '171220', '171227', '180103']

    classNum = -1
    className = []
    classNumAmount = []

    for path in train_folder:
        x_data = []
        y_data = []
        x_test = []
        y_test = []
        classdir = '/HDD/dataset/ITRI LeafClassification/images_resized_500_' + path + '/'
        #recursive read image data
        for dirPath, dirNames, fileNames in os.walk(classdir):
            
            jpgDataFound = 0

            # how much jpg data in the folder
            fileNum = 0
            for jpg in fileNames:
                img_path = os.path.join(dirPath, jpg)
                if jpg.endswith(".jpg"):
                    jpgDataFound = 1
                    fileNum += 1
                    
            ## random select validation index
            if jpgDataFound == 1:
                classNumAmount.append(fileNum)
                classNum += 1
                className.append(dirPath[len(classdir):len(dirPath)])
                validation_num = round(fileNum * validation_rate)
                validation_index = np.random.choice(range(fileNum), validation_num, replace = False)
                # print(validation_num)
            
            jpg_file_index = 0
            validation_data_num = 0
            train_data_num = 0

            for jpg in fileNames:
                img_path = os.path.join(dirPath, jpg)
                # generate train data
                if jpg.endswith(".jpg") and (jpg_file_index not in validation_index):   
                    x_img = scipy.misc.imread(img_path)
                    x_img = scipy.misc.imresize(x_img, [resolute, resolute])
                    x_img = np.array(x_img)
                    x_data.append(x_img)
                    y_data.append(classNum)
                    train_data_num += 1
                    jpg_file_index += 1

                # generate validation data
                elif jpg.endswith(".jpg") and (jpg_file_index in validation_index):
                    x_img = scipy.misc.imread(img_path)
                    x_img = scipy.misc.imresize(x_img, [resolute, resolute])
                    x_img = np.array(x_img)
                    x_test.append(x_img)
                    y_test.append(classNum)
                    validation_data_num += 1
                    jpg_file_index += 1

            if jpgDataFound == 1:
                # print(fileNum)
                print('Type {} - {} : {} datas, [{}, {}]'.format(classNum, dirPath[len(classdir):len(dirPath)], fileNum, train_data_num, validation_data_num))

        x_test = np.array(x_test)
        y_test = np.array(y_test)  
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        write_pickle = (x_data, y_data, x_test, y_test)
        print('dumping pickle')

        pickleDir = '/HDD/joshua/pickle_demo/data' + path + '.pickle'

        with open(pickleDir, 'wb') as p:
            pickle.dump(write_pickle, p, protocol = 4)
        print('done pickle : ' + path)


    classNumAmount = np.array(classNumAmount)

    return className, classNumAmount, classNum


def main():

    className, classNumAmount, classNum = read_data()
    with open('/HDD/joshua/pickle_demo/dataIndex.csv', 'w', encoding='utf8') as csvfile:
        for i in range(len(className)):
            csvfile.write('{}, {}, {}\n'.format(i, className[i], classNumAmount[i]))  
    print('done data index')

if __name__ == '__main__':
    main()

