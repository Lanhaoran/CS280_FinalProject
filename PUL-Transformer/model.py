import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import math

from Getdata import MyDataset
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.DoubleTensor)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=64):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)    #64*512
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    #64*1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))    #256   model/2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)   #64*1*512

    def forward(self, x):     #[seq,batch,d_model]
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        return x + self.pe[:x.size(0), :]   #64*64*512


class MLP(torch.nn.Module):
    def __init__(self,num_i,num_h1, num_h2, num_o, dropout =0):
        super(MLP,self).__init__()
        
        self.linear1=nn.Linear(num_i,num_h1)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(num_h1,num_h2) #2个隐层
        self.relu2=nn.ReLU()
        self.linear3=nn.Linear(num_h2,num_o)
        self.dropout = nn.Dropout(p = dropout)
    
    def init_weights(self):
        initrange = 0.1
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear3.bias.data.zero_()
        self.linear3.weight.data.uniform_(-initrange, initrange)
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class TransAm(nn.Module):
    def __init__(self, feature_size, out_size, seq_length, d_model=512, num_layers=1, dropout=0):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.embedding=nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_length)          #50*512
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = 8, dropout = dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)
        self.decoder = MLP(d_model, 256, 128, out_size, dropout)
        self.decoder.init_weights()
        self.src_key_padding_mask = None

    '''
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    '''

    def forward(self, src):
        #shape of src  [seq, batch, feature_size]
        '''
        if self.src_key_padding_mask is None:
            mask_key = src_padding   #[batch,seq]
            self.src_key_padding_mask = mask_key
        '''
        src = self.embedding(src)        #[seq, batch, d_model]
        src = self.pos_encoder(src)    #[seq, batch, d_model]
        output = self.transformer_encoder(src, self.src_mask, self.src_key_padding_mask)  # , self.src_mask)
        decoder_input = output[0]   # [batch, d_model]  使用seq第一个向量作分类
        output = self.decoder(decoder_input)  #[batch,output_size]
        self.src_key_padding_mask = None
        return output
    
    def encode_out(self, src):
        src = self.embedding(src)        #[seq, batch, d_model]
        src = self.pos_encoder(src)    #[seq, batch, d_model]
        output = self.transformer_encoder(src, self.src_mask, self.src_key_padding_mask)  # , self.src_mask)

        return output


if __name__ == '__main__':

    #####################################
    ## Get data
    #####################################
    dataset_name = 'Beef'
    datapath = r'E:\Character Motion\Prompt_cluster\DTCR_Ip\UCRArchive_2018\{0}\{0}_TRAIN.tsv'.format(dataset_name)
    datapath_motion = r'E:\Character Motion\Prompt_cluster\dataset\dataset\data\train_data_1.npy'
    dataset = MyDataset(datapath, 'UCR')
    #dataset = MyDataset(datapath_motion, 'Motion')
    print("Dataset shape: {}".format(dataset.data.shape))
    

    #######################################
    ## Model
    ######################################
    seq_length = dataset.data.shape[1]
    feature_size = dataset.data.shape[2]
    out_size = 5     # reconstruction
    d_model = 128     
    model = TransAm(feature_size, out_size, seq_length, d_model, dropout=0.15)
    
    train_loader = DataLoader(dataset=dataset, batch_size= 30, shuffle=True)
    for data, labels in train_loader:
        # data: [batch_size, seq_length, feature_size]
        # label: [batch_size,]
        enc_inputs = data.permute([1,0,2])     # [seq_length, batch_size, feature_size]
        #key_padding_mask = torch.ones(enc_inputs.shape[1], enc_inputs.shape[0])  # [batch_size, seq_length]
        output = model.forward(enc_inputs)
        enc_out = model.encode_out(enc_inputs)
        model.eval()
        print("Model output size: {}".format(output.shape))
        print("Encoder output size: {}".format(enc_out.shape))

