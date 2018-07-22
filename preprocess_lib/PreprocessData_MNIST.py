import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def read_images_data(in_file_path):
    #
    with open(in_file_path, 'rb') as f_id:
        f_id.seek(4,0)
        num_images = struct.unpack('>I', f_id.read(4))[0]
        image_height = struct.unpack('>I', f_id.read(4))[0]
        image_width = struct.unpack('>I', f_id.read(4))[0]
        i = 0
        #
        img_data = np.ndarray([num_images,image_height,image_width,1],
                              dtype = np.uint32)
        samples =  np.ndarray([num_images,image_height * image_width],
                              dtype = np.uint32)
        while i < num_images:
            tmp_data = np.array(struct.unpack(">784B", f_id.read(784)))
            tmp_data =  np.flipud(tmp_data.reshape([image_height,image_width]))
            samples[i] = tmp_data.reshape([1,image_height * image_width])
            #tmp_data = tmp_data[np.newaxis]
            img_data[i] = tmp_data.reshape([image_height,image_width,1])
            i += 1
        #
        return img_data,samples

def read_labels_data(in_file_path):
    #
    labels = []
    with open(in_file_path, 'rb') as f_id:
        f_id.seek(4,0)
        num_labels = struct.unpack('>I', f_id.read(4))[0]
        i = 0
        #
        while i < num_labels:
            tmp_label = struct.unpack(">B", f_id.read(1))[0]
            labels.append(tmp_label)
            i += 1
        #
        return labels

def one_hot_labels(labels):
    #
    data = []
    for tmo_label in labels:
        one_hot_label = [0] * 10
        one_hot_label[tmo_label] = 1
        data.append(one_hot_label)
    return np.array(data)

def view_data(train_data, train_labels, num_data):
    #
    for i in range(num_data):
        img_data = train_data[i].reshape([train_data[i].shape[0],train_data[i].shape[1]])
        print(train_labels[i])
        plt.figure()
        # plt.ion()
        plt.imshow(img_data, interpolation = "nearest", cmap = "bone", origin = "low")
        plt.show()


def view_data_distribution(train_data, test_data):
    #
    num_train_data = [train_data.count(0),train_data.count(1),train_data.count(2),
                      train_data.count(3),train_data.count(4),train_data.count(5),
                      train_data.count(6),train_data.count(7),train_data.count(8),
                      train_data.count(9)]
    num_test_data = [test_data.count(0),test_data.count(1),test_data.count(2),
                     test_data.count(3),test_data.count(4),test_data.count(5),
                     test_data.count(6),test_data.count(7),test_data.count(8),
                     test_data.count(9)]
    x_axis = (0,1,2,3,4,5,6,7,8,9)
    #
    #
    plt.figure(figsize=(15,5))
    #
    plt.subplot(1,2,1)
    plt.xlim(0,10)
    plt.ylim(0,max(num_train_data))
    plt.ylabel("label's num")
    plt.bar(x_axis,num_train_data)
    #
    plt.subplot(1,2,2)
    plt.xlim(0,10)
    plt.ylim(0,max(num_test_data))
    plt.ylabel("label's num")
    plt.bar(x_axis,num_test_data)
    #
    plt.show()

def train_data_iterator(samples, labels, iteration_steps, chunkSize):
    """
    Iterator/Generator: get a batch of data
    这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
    用于 for loop， just like range() function
    """
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    stepStart = 0  # initial step
    i = 0
    while i < iteration_steps:
        stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
        yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
        i += 1


def test_data_iterator(samples, labels, chunkSize):
    """
    Iterator/Generator: get a batch of data
    这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
    用于 for loop， just like range() function
    """
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    stepStart = 0  # initial step
    i = 0
    while stepStart < len(samples):
        stepEnd = stepStart + chunkSize
        if stepEnd < len(samples):
            yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
            i += 1
        stepStart = stepEnd

current_dir = os.getcwd()
train_images,train_samples = read_images_data(current_dir + r"\Data\MNIST\train-images.idx3-ubyte")
train_labels = read_labels_data(current_dir + r"\Data\MNIST\train-labels.idx1-ubyte")
dl_train_labels = one_hot_labels(train_labels)
#
test_images,test_samples = read_images_data(current_dir + r"\Data\MNIST\t10k-images.idx3-ubyte")
test_labels = read_labels_data(current_dir + r"\Data\MNIST\t10k-labels.idx1-ubyte")
dl_test_labels = one_hot_labels(test_labels)
#
image_height = 28
image_width = 28
num_channels = 1
num_labels = 10
#
# view_data_distribution(train_labels, test_labels)
# view_data(train_images, dl_train_labels, 3)
