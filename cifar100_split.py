import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from PIL import Image


class Cifar100Split(Dataset):
    def __init__(self, file_path="data/cifar-100-python/", start_num=0, end_num=10, train=True, one_hot=False,
                 transform=None,label_align=True):
        self.transform = transform
        self.label_align=label_align
        if train is True:
            self.data = self.create_split_train(file_path=file_path, start_num=start_num, end_num=end_num,
                                                one_hot=one_hot)
        else:
            self.data = self.create_split_test(file_path=file_path, start_num=start_num, end_num=end_num,
                                               one_hot=one_hot)
        self.x = self.data['x']
        self.y = self.data['y'].tolist()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def create_split_train(self, file_path="data/cifar-100-python/", start_num=0, end_num=10, one_hot=False):
        data_dict = self.load_cifar100(file_path=file_path, one_hot=False)
        x_train = data_dict['x_train']
        y_train = data_dict['y_train']
        y_train = np.array(y_train)
        num_class = end_num - start_num
        a1 = y_train >= start_num
        a2 = y_train < end_num
        index = a1 & a2
        task_train_x = x_train[index]
        label = y_train[index]
        if self.label_align:
            label = label - start_num
        if one_hot is True:
            task_train_y = np.zeros([task_train_x.shape[0], num_class])
            task_train_y[range(task_train_x.shape[0]), label] = 1
        else:
            task_train_y = label

        task_split = {
            "x": task_train_x,
            "y": task_train_y,
        }

        return task_split

    def create_split_test(self, file_path="data/cifar-100-python/", start_num=0, end_num=10, one_hot=False):
        data_dict = self.load_cifar100(file_path=file_path, one_hot=False)

        x_test = data_dict['x_test']
        y_test = data_dict['y_test']
        y_test = np.array(y_test)
        num_class = end_num - start_num

        a1 = y_test >= start_num
        a2 = y_test < end_num
        index = a1 & a2
        task_test_x = x_test[index]
        label = y_test[index]
        if self.label_align:
            label = label - start_num
        if one_hot is True:
            task_test_y = np.zeros([task_test_x.shape[0], num_class])
            task_test_y[range(task_test_x.shape[0]), label] = 1
        else:
            task_test_y = label
        task_split = {
            "x": task_test_x,
            "y": task_test_y
        }

        return task_split

    def load_cifar100(self, file_path="data/cifar-100-python/", one_hot=True):
        # 读取训练数据
        with open(file_path + "train", 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_train = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_temp = dict[b'fine_labels']
        if one_hot is True:
            y_train = np.zeros([x_train.shape[0], 100])
            y_train[range(x_train.shape[0]), y_temp] = 1
        else:
            y_train = y_temp
        # 读取测试数据
        with open(file_path + "test", 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_test = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_temp = dict[b'fine_labels']
        if one_hot is True:
            y_test = np.zeros([x_test.shape[0], 100])
            y_test[range(x_test.shape[0]), y_temp] = 1
        else:
            y_test = y_temp

        # x_train, x_test = normalize(x_train, x_test)
        data = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test
        }
        # print(dict[b'data'].shape)
        # print(dict[b'fine_labels'])
        # print(dict[b'coarse_labels'])
        # show(a[20000])
        return data



