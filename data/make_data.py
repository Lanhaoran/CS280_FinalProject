import numpy as np
import sys
import os
import torch
from torch.utils.data import random_split

# 调整股票数据集，划分出训练集与测试集

end_price_path = r'C:\Users\DELL\Desktop\github\data\end_price.npy'
multi_dim_path = r'C:\Users\DELL\Desktop\github\data\multi_dim.npy'

end_prcie_matrix = np.load(end_price_path, allow_pickle=True)
num_stocks, time_steps = end_prcie_matrix.shape  # 300,107
end_prcie_matrix =  np.expand_dims(end_prcie_matrix, axis=2) # (300,107,1)


multi_dim_matrix = np.load(multi_dim_path, allow_pickle=True)
num_stocks, time_steps, num_features = multi_dim_matrix.shape  # 300,107,11

torch.manual_seed(0)
end_price_train, end_price_test = random_split( 
                                dataset = end_prcie_matrix, lengths=[200,100])
multi_dim_train, multi_dim_test = random_split( 
                                dataset = multi_dim_matrix, lengths=[200,100])

# save
np.save("end_price_TRAIN.npy", np.array(list(end_price_train)))
np.save("end_price_TEST.npy", np.array(list(end_price_test)))
np.save("multi_dim_TRAIN.npy", np.array(list(multi_dim_train)))
np.save("multi_dim_TEST.npy", np.array(list(multi_dim_test)))

