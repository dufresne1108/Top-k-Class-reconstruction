
import torchvision

from torch.utils.data import Dataset, DataLoader
import numpy as np

from PIL import Image


class Cifar10_SVNH_Split(Dataset):
    def __init__(self, file_path="data/", isCifar10=True, start_num=0, end_num=10, train=True, one_hot=False,
                 transform=None, label_align=True):
        self.transform = transform
        self.label_align = label_align
        self.data = self.create_split_data(file_path=file_path, train=train, isCifar10=isCifar10,
                                           start_num=start_num, end_num=end_num, one_hot=one_hot)
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

    def create_split_data(self, file_path="/data", train=True, isCifar10=True, start_num=0, end_num=10, one_hot=False):
        if isCifar10:
            train_data = torchvision.datasets.CIFAR10(
                root=file_path,
                train=train,
                download=True,
            )
            x_train = train_data.data
            y_train = train_data.targets
            y_train = np.array(y_train)
        else:
            if train:
                mode = "train"
            else:
                mode = "test"
            train_data = torchvision.datasets.SVHN(
                root=file_path,
                split=mode,
                download=True,
            )
            x_train = train_data.data.transpose(0, 2, 3, 1)
            y_train = train_data.labels

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


class Cifar10_SVNH_Rehearsal(Dataset):
    def __init__(self, file_path="data/", isCifar10=True, end_num=10, rehearsal_size=2000,
                 train=True, transform=None, label_align=True):
        self.transform = transform
        self.label_align = label_align
        self.transform = transform
        self.label_align = label_align
        self.rehearsal_size = rehearsal_size
        self.end_num = end_num
        self.num_per_class = self.rehearsal_size // self.end_num
        self.data = self.produce_rehearsal(file_path=file_path, train=train, isCifar10=isCifar10)
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

    def produce_rehearsal(self, file_path="/data", train=True, isCifar10=True):
        if isCifar10:
            train_data = torchvision.datasets.CIFAR10(
                root=file_path,
                train=train,
                download=True,
            )
            x_train = train_data.data
            y_train = train_data.targets
            y_train = np.array(y_train)
        else:
            if train:
                mode = "train"
            else:
                mode = "test"
            train_data = torchvision.datasets.SVHN(
                root=file_path,
                split=mode,
                download=True,
            )
            x_train = train_data.data.transpose(0, 2, 3, 1)
            y_train = train_data.labels

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
