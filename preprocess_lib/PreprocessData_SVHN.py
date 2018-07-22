import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

#
def one_hot_labels(labels):
    labels = np.array([x[0] for x in labels])
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0]*10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return labels

def normalize_data(samples):
    #图片灰度化
    #0 - 255 -> -1.0 - +1.0
    samples = np.add.reduce(samples, keepdims = True, axis = 3) #R + G + B
    samples = samples / 3.0
    return samples / 128.0 - 1.0

def read_data(train_data_path, test_data_path):
    train_data = sio.loadmat(train_data_path)
    test_data = sio.loadmat(test_data_path)
    #
    image_height = len(train_data['X'][:, 0, 0, 0])
    image_width = len(train_data['X'][0,:,0,0])
    num_channels = 1
    num_images = len(train_data['X'][0, 0, 0, :])
    num_labels = 10
    #
    tmp_train_samples = np.transpose(train_data['X'], (3, 0, 1, 2))
    tmp_test_samples = np.transpose(test_data['X'], (3, 0, 1, 2))
    train_samples = normalize_data(tmp_train_samples)
    train_labels = one_hot_labels(train_data['y'])
    test_samples = normalize_data(tmp_test_samples)
    test_labels = one_hot_labels(test_data['y'])

    return (train_samples, train_labels, test_samples, test_labels)

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

num_labels = 10
image_size = 32
num_channels = 1

if __name__ == "__main__":
    #
    pass
    #
    # distribution(train_data['y'], 'Train Labels')
    # distribution(test_data['y'], 'Test Labels')
