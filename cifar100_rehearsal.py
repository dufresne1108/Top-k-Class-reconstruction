import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from PIL import Image


class Cifar100Rehearsal(Dataset):
    def __init__(self, file_path="data/cifar-100-python/", end_num=10, rehearsal_size=2000, transform=None,
                 label_align=True):
        self.transform = transform
        self.label_align = label_align
        self.rehearsal_size = rehearsal_size
        self.end_num = end_num
        self.num_per_class = self.rehearsal_size // self.end_num
        self.data = self.produce_rehearsal(file_path=file_path)
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

    def produce_rehearsal(self, file_path="data/cifar-100-python/"):
        data_dict = self.load_cifar100(file_path=file_path, one_hot=False)
        x_train = data_dict['x_train']
        y_train = data_dict['y_train']
        y_train = np.array(y_train)
        rehearsal_data = None
        rehearsal_label = None

        for i in range(1, self.end_num):
            a1 = y_train >= i - 1
            a2 = y_train < i
            index = a1 & a2
            task_train_x = x_train[index]
            label = y_train[index]
            index = np.random.choice(task_train_x.shape[0], self.num_per_class)
            tem_data = task_train_x[index]
            tem_label = label[index]
            if rehearsal_data is None:
                rehearsal_data = tem_data
                rehearsal_label = tem_label
            else:
                rehearsal_data = np.concatenate([rehearsal_data, tem_data], axis=0)
                rehearsal_label = np.concatenate([rehearsal_label, tem_label], axis=0)

        task_split = {
            "x": rehearsal_data,
            "y": rehearsal_label,
        }
        return task_split
