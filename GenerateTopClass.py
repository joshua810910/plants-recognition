import os
import numpy as np
import pickle
import csv

def read_ITRI_data(path, name):
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    pickle_path = path 
    pickle_list = ["171107", "171113", "171121", "171128", "171204", "171211", "171220", "171227", "180103"]

    for date in pickle_list:
        x_data, x_label, y_data, y_label = read_pickle(pickle_path, name + date + ".pickle")

        x_data = np.array(x_data)
        x_label = np.array(x_label)
        y_data = np.array(y_data)
        y_label = np.array(y_label)

        for i in range(np.shape(x_data)[0]):
            train_data.append(x_data[i])
            train_label.append(x_label[i])
        for i in range(np.shape(y_data)[0]):
            test_data.append(y_data[i])
            test_label.append(y_label[i])

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    return train_data, train_label, test_data, test_label

def selected_data(x_data, x_label, class_index):
    sl_label = []
    sl_index = []
    
    for i in range(np.shape(x_label)[0]):
        if x_label[i] in class_index:
            sl_label.append(x_label[i])
            sl_index.append(i)
    sl_data = x_data[sl_index]
    return np.array(sl_data), np.array(sl_label)

def relabel(x_label, class_index):
    for i in range(len(x_label)):
        x_label[i] = list(class_index).index(x_label[i])

    # flag = []
    # for i in range(len(x_label)):
    #     if x_label[i] not in flag:
    #         flag.append(x_label[i])
    #         x_label[i] = flag.index(x_label[i])
    #     else:
    #         x_label[i] = flag.index(x_label[i])

    return x_label

def read_pickle(pickle_path, pickle_file):

    print("Reading %s" % (pickle_path + '/' + pickle_file))

    with open(os.path.join(pickle_path, pickle_file), 'rb') as p:
        pickled_data = pickle.load(p)
    train_data = pickled_data[0]
    train_label = pickled_data[1]
    test_data = pickled_data[2]
    test_label = pickled_data[3]

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    return train_data, train_label, test_data, test_label

def split_ITRI_data(to_write_pickle, target_class_num, pickle_file_folder):

    class_name = []
    with open('/HDD/joshua/pickle_demo/dataIndex.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # print(row[1])
            class_name.append(row[1])

    train_data, train_label, test_data, test_label = read_ITRI_data("/HDD/joshua/pickle_demo", 'data')

    total_class_num = np.max(train_label) + 1

    # select top n numbers class
    class_amount = np.zeros(total_class_num)
    for data in train_label:
        class_amount[data] += 1

    ## to choose from range
    # class_index = []
    # for i in range(np.shape(class_amount)[0]):
    #     if class_amount[i] >= 100 and class_amount[i] <= 180:
    #         class_index.append(i)

    ## to choose top n class
    class_index = np.argsort(class_amount)
    class_index = np.flip(class_index, axis = 0)
    class_index = class_index[0:target_class_num]

    print(class_amount[class_index])

    train_data, train_label = selected_data(train_data, train_label, class_index)
    test_data, test_label = selected_data(test_data, test_label, class_index)

    train_label = relabel(train_label, class_index)
    test_label = relabel(test_label, class_index)

    # write pickle file
    if to_write_pickle == True:
        if os.path.exists(pickle_file_folder) == False:
            os.mkdir(pickle_file_folder)
        write_pickle = (train_data, train_label, test_data, test_label)
        
        print('dumping pickle')

        with open(os.path.join(pickle_file_folder, 'data.pickle'), 'wb') as p:
            pickle.dump(write_pickle, p, protocol = 4)
        print('pickle data done')

        with open(os.path.join(pickle_file_folder, 'data.txt'), 'w') as f:
            f.write('original index, {}'.format(class_index[0]))
            for index in class_index[1:len(class_index)]:
                f.write(', {}'.format(index))
            f.write('\nnumber of each, {}'.format(class_amount[class_index[0]]))
            for index in class_index[1:len(class_index)]:
                f.write(', {}'.format(int(class_amount[index])))

        with open(os.path.join(pickle_file_folder, 'dataIndex.csv'), 'w', encoding='utf8') as csvfile:
            for i in range(target_class_num):
                csvfile.write('{}, {}\n'.format(i, class_name[class_index[i]]))  
        print('done data index')
        

    return train_data, train_label, test_data, test_label, len(class_index)


def main():

    target_class_num = 2000
    
    read_new_data = True
    to_write_pickle = True

    print('generating top {} classes...'.format(target_class_num))

    if read_new_data:
        train_data, train_label, test_data, test_label, target_class_num = split_ITRI_data(to_write_pickle, target_class_num, "/HDD/joshua/DemoTop" + str(target_class_num))
        
if __name__ == '__main__':
    main()
