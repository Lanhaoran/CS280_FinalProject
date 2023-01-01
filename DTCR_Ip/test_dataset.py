import os
import warnings
warnings.filterwarnings('ignore')

#import tensorflow as tf
import numpy as np
import random
from utils import read_dataset, construct_classification_dataset, show_train_test_curve
from config import config_dtcr
from framework import DTCR

NORMALIZED = True
'''dataset setting'''
def read_dataset_motion(opt, dataset_type, label_dict=None, if_n=False):
    '''
    normal_cluster: 代表正常的标签，在所有数据集中，将数据占比多的一方视为正常数据
    split: 分割数据的段数
    '''
    if dataset_type == 'train':
        data = np.load(opt['motion_train_file'], allow_pickle=True)
    elif dataset_type == 'test':
        data = np.load(opt['motion_test_file'], allow_pickle=True)

    #label = data[:,0]
    label = np.array([l[0] for l in data[:,0]])   # (N_train, 96)
    label = label + 10
    label = -1 * label
    
    if label_dict is None:
        label_dict = {}
        label_list = np.unique(label)
        for idx in range(len(label_list)):
            label_dict[str(label_list[idx])] = idx#key：-1*原始label，value：新label

    o_label = list(label_dict.keys())
    for l in o_label:
        label[label == float(l)] = label_dict[l]
        
    label = label.astype(int)
    data = data[:,1:]

    #----------------------------------------
    '''
    if dataset_type == 'test' and 'MIT' in opts['test_file']:
        tmp_data = []
        tmp_label = []
        for item in np.unique(label):
            tmp_data.append(data[label == item][0:50])
            tmp_label.append(label[label == item][0:50])
        data = np.concatenate(tmp_data, axis=0)
        label = np.concatenate(tmp_label, axis=0)
    '''    
    #----------------------------------------
    if if_n == True:
        for i in range(data.shape[0]):
            data[i] = normalize(data[i])

    
    #数据集中的类别数量
    print(dataset_type)       
    print('Number of class: ', len(np.unique(label)))
    print('Number of sample:', data.shape[0])
    print('Time Series Length: ', data.shape[1])
    if data.ndim > 2: 
        print('Feature dimension per time step: ', data.shape[2])
        config_dtcr['feature_num'] = data.shape[2]
    else:
        print('Feature dimension per time step: 1')
        config_dtcr['feature_num'] = 1
    return data, label, label_dict

def normalize(seq):
    return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq)) - 1

# data_path = 'E:\Character Motion\Prompt_cluster\dataset\dataset\data.npy'
# read_dataset_motion(config_dtcr, 'test', if_n=NORMALIZED)