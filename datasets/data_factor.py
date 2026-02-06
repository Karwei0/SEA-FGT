import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MSL_data_loader(Dataset):
    def __init__(self, root_data_path, win_size, step, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_data_path, 'MSL_train.npy')
        data = np.load(train_path)

        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_path = os.path.join(root_data_path, 'MSL_test.npy')
        test_data = np.load(test_path)

        self.train = data
        self.test = self.scaler.transform(test_data)
        self.val = self.test
        self.test_label = np.load(os.path.join(root_data_path, 'MSL_test_label.npy'))

    def __len__(self):
        if self.mode == 'train':
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index *= self.step
        if self.mode == 'train':
            return np.float32(self.train[index: index + self.win_size]), np.float32(
                self.test_label[index: index + self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index: index + self.win_size]), np.float32(
                self.test_label[index: index + self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index: index + self.win_size]), np.float32(
                self.test_label[index: index + self.win_size])
        return np.float32(self.test[
                          index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]), np.float(
            self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]
        )


class PSM_data_loader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class SMAP_data_loader(Dataset):
    def __init__(self, root_data_path, win_size, step, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(os.path.join(root_data_path, 'SMAP_train.npy'))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = np.load(os.path.join(root_data_path, 'SMAP_test.npy'))
        self.train = data
        self.test = test_data
        self.val = test_data
        self.test_label = np.load(os.path.join(root_data_path, 'SMAP_test_label.npy'))

    def __len__(self):
        if self.mode == 'train':
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index *= self.step
        if self.mode == 'train':
            return np.float32(self.train[index: index + self.win_size]), np.float32(
                self.test_label[index: index + self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index: index + self.win_size]), np.float32(
                self.test_label[index: index + self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index: index + self.win_size]), np.float32(
                self.test_label[index: index + self.win_size])
        return np.float32(self.test[
                          index // self.step * self.win_size: index // self.step * self.win_size * self.win_size + self.win_size]), np.float32(
            self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]
        )


class SMD_data_loader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")[:,:]
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")[:,:]
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")[:]

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWAT_data_loader(Dataset):
    def __init__(self, root_data_path, win_size, step, mode='train'):
        self.mode = mode
        self.win_size = win_size
        self.step = step
        self.scaler = StandardScaler()
        
        train_data = pd.read_csv(os.path.join(root_data_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_data_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        self.val = self.test
        self.test_label = labels

    def __len__(self):
        if self.mode == 'train':
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_label[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_label[index:index+self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_label[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_label[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

