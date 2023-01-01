import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch


# 定义dataset类
class MyDataset(Dataset):
    # 初始化函数，得到数据
    def __init__(self, path, d_type):
        self.data_label = None
        self.data = None
        self.label = None

        if d_type == 'Motion': 
            self.data_label = np.load(path, allow_pickle=True)
            self.label = self.data_label[:,0,0].astype(int)
            self.data = self.data_label[:,1:]
        else: 
            self.data_label = np.loadtxt(path) 
            self.label = self.data_label[:,0].astype(int)
            self.data = self.data_label[:,1:]
            self.data = np.expand_dims(self.data, axis = 2)
        

        self.mean = torch.mean(torch.from_numpy(self.data))
        self.std = torch.std(torch.from_numpy(self.data))
        self.max = torch.max(torch.from_numpy(self.data))
        self.min = torch.min(torch.from_numpy(self.data))
 
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
 
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

    def set_data(self, data, label):
        self.data = data
        self.label = label


    def standardization(self):
        self.data = (self.data - self.mean.numpy()) / self.std.numpy()

    def max_min_noralize(self):
        self.data = (self.data - self.min.numpy()) / (self.max.numpy() - self.min.numpy())

if __name__ == '__main__':

    dataset_name = 'Beef'
    datapath = r'E:\Character Motion\Prompt_cluster\DTCR_Ip\UCRArchive_2018\{0}\{0}_TRAIN.tsv'.format(dataset_name)
    datapath_motion = r'E:\Character Motion\Prompt_cluster\dataset\dataset\data\train_data_1.npy'
    dataset = MyDataset(datapath, 'UCR')
    #dataset = MyDataset(datapath_motion, 'Motion')
    print(dataset.data.shape)
    train_loader = DataLoader(dataset=dataset, batch_size= 8, shuffle=True)
    for data, labels in train_loader:
        print(data.shape)
        print(labels.shape)





