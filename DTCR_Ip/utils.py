import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from scipy.special import comb
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from config import config_dtcr

def normalize(seq):
    return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq)) - 1

def read_dataset(opts, dataset_type, label_dict=None, if_n=False):
    '''
    normal_cluster: 代表正常的标签，在所有数据集中，将数据占比多的一方视为正常数据
    split: 分割数据的段数
    '''
    if dataset_type == 'train':
        data = np.loadtxt(opts['train_file'])
    elif dataset_type == 'test':
        data = np.loadtxt(opts['test_file'])
    elif dataset_type == 'v':
        data = np.loadtxt(opts['test_file'])
    
            
    label = data[:,0]
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
    
    if dataset_type == 'test' and 'MIT' in opts['test_file']:
        tmp_data = []
        tmp_label = []
        for item in np.unique(label):
            tmp_data.append(data[label == item][0:50])
            tmp_label.append(label[label == item][0:50])
        data = np.concatenate(tmp_data, axis=0)
        label = np.concatenate(tmp_label, axis=0)
        
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

def shuffle_timeseries(data, rate=0.2):
    # 打乱一定比率的数据
    ordered_index = np.arange(len(data))
    ordered_index.astype(int)
    # 选定要打乱的index
    shuffled_index = np.random.choice(ordered_index, size=int(np.floor(rate*len(data))), replace=False)
    ordered_index[shuffled_index] = -1
    # 打乱
    shuffled_index = np.random.permutation(shuffled_index)
    ordered_index[ordered_index == -1] = shuffled_index
    data = data[ordered_index]
    
    return data

def construct_classification_dataset(dataset):
    # 张量转化为ndarray
    # session = tf.Session()
    #real_dataset = dataset.eval(session=session)
    #session.close()
    real_dataset = copy.deepcopy(dataset)
    fake_dataset = []
    for seq in real_dataset:
        fake_dataset.append(shuffle_timeseries(seq))
    fake_dataset = np.array(fake_dataset)
    
    label = np.array([1]*fake_dataset.shape[0] + [0]*real_dataset.shape[0])
    dataset = np.concatenate([fake_dataset, real_dataset], axis=0)
    
    label = np.random.permutation(label)
    dataset = np.random.permutation(dataset)
    
    # print('dataset shape: ', dataset.shape)
    # print('label shape:', label.shape)
    
    return dataset, label

def truncatedSVD(matrix, K):
    svd = TruncatedSVD(n_components=K)
    truncated_matrix = svd.fit_transform(matrix)
    return truncated_matrix
    

def ri_score(y_true, y_pred):
    print("y_true: {}".format(y_true))
    print("y_pred: {}".format(y_pred))
    tp_plus_fp = comb(np.bincount(y_true), 2).sum()
    tp_plus_fn = comb(np.bincount(y_pred), 2).sum()
    A = np.c_[(y_true, y_pred)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(y_true))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def nmi_score(y_true, y_pred):
    #print("y_true: {}".format(y_true))
    #print("y_pred: {}".format(y_pred))
    return normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')

def cluster_using_kmeans(embeddings, K):
    return KMeans(n_clusters=K).fit_predict(embeddings)

def show_train_test_curve(opts, train, test, index='', batch_i = None):        
    file_name = '{}__{}_en_{}_lambda_{}_train_curve.png'.format(index, batch_i, opts['indicator'], opts['encoder_hidden_units'], opts['lambda'])

    x = np.arange(len(train))
    x *= opts['test_every_epoch']
    
    plt.plot(x, train, label='train')
    plt.plot(x, test, label='test')
    plt.title('{} curve '.format(opts['indicator']))
    plt.xlabel('epoch')
    plt.ylabel(opts['indicator'])
    plt.ylim((0,1))
    plt.legend()
    plt.savefig(opts['img_path']+'/'+file_name)
    plt.close()

def show_loss_curve(opts, loss_list, index='', batch_i = None):
    loss_names = ['complete_loss', 'recons_loss', 'cls_loss', 'kmeans_loss']
    for id in range(len(loss_names)):  
        file_name = '{}__en_{}_lambda_{}_{}_curve.png'.format(index, opts['encoder_hidden_units'], opts['lambda'], loss_names[id])
        x = np.arange(opts['max_iter'])
        #x *= opts['test_every_epoch']
        plt.plot(x, loss_list[id], label=loss_names[id])
        # plt.plot(x, loss_list[1], label='recons_loss')
        # plt.plot(x, loss_list[2], label='cls_loss')
        # plt.plot(x, loss_list[3], label='kmeans_loss')
        plt.title('{} curve (en_{}, lambda_{})'.format(loss_names[id], opts['encoder_hidden_units'], opts['lambda']))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        #plt.ylim((0,1))
        plt.legend()
        path = os.path.join(opts['img_path'], loss_names[id])
        if os.path.exists(path) == False: os.makedirs(path)
        plt.savefig(path+'/'+file_name)
        plt.close()

def get_Batch(data, label, opts):
    # 把data分为许多个batch [(batch_x,batch_y) ... ]
    batch_size = opts['batch_size']
    n_batches = opts['training_samples_num'] // batch_size 
    train_sample_num = n_batches*batch_size
    if opts['feature_num'] == 1: 
        batches_data = np.array_split(data[:train_sample_num,:], n_batches, axis=0)
    else:
        batches_data = np.array_split(data[:train_sample_num,:,:], n_batches, axis=0)
    batches_label = np.array_split(label[:train_sample_num,], n_batches, axis=0)
    cls_train_data =[]
    cls_train_label = []
    for batch_data in batches_data:
        cls_train_data_batch, cls_train_label_batch = construct_classification_dataset(batch_data)
        cls_train_data.append(cls_train_data_batch)
        cls_train_label.append(cls_train_label_batch)
    print("Iteration: {}, batch size: {}, sample number: {}/{}".format(n_batches, batch_size, train_sample_num, opts['training_samples_num']))
    return cls_train_data, cls_train_label, batches_data, batches_label, n_batches
    