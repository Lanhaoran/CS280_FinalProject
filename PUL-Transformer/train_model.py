import os
import math
import sys
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from typing import Dict
from Getdata import MyDataset
from torch.utils.data import DataLoader
from model import TransAm
import torch.optim as optim
from tqdm import tqdm
from Scheduler import GradualWarmupScheduler

from sklearn.cluster import KMeans

from torch.utils.tensorboard import SummaryWriter


def evaluate_accuracy(X, y_true, model):
    '''
    model_output : [batch, label_size]
    y_true: [batch,]
    '''
    with torch.no_grad():
        model.eval()
        n = y_true.shape[0]
        enc_inputs = X.permute([1,0,2])     # [seq_length, batch_size, feature_size]
        output = model.forward(enc_inputs)
        y_pred = np.argmax(output.cpu(), axis = 1)
        correct = float(sum(y_pred[i] == y_true[i] for i in range(n)))
    
    return float(correct / n)

def eval_acc(data_, label_, model_):
    data = data_
    labels = label_
    model = model_
    with torch.no_grad():
        acc = evaluate_accuracy(data, labels, model)
    return acc


def GetDataSet(modelConfig):
    dataset_name = modelConfig["data"]
    if dataset_name == 'Motion':
        train_datapath = r'E:\Character Motion\Prompt_cluster\dataset\dataset\data\train_data_1.npy'
        test_datapath = None
    else:
        train_datapath = r'E:\Character Motion\Prompt_cluster\DTCR_Ip\UCRArchive_2018\{0}\{0}_TRAIN.tsv'.format(dataset_name)
        test_datapath = r'E:\Character Motion\Prompt_cluster\DTCR_Ip\UCRArchive_2018\{0}\{0}_TEST.tsv'.format(dataset_name)
    
    train = MyDataset(train_datapath, dataset_name)
    print("Training data shape: {}, training label shape: {}".format(train.data.shape, train.label.shape))

    test = MyDataset(test_datapath, dataset_name)
    print("Test data shape: {}, test label shape: {}".format(test.data.shape, test.label.shape))

    return train, test

def step(data, labels, test_data, test_label, model, optimizer, criterion, device, modelConfig):
    enc_inputs = data.permute([1,0,2])     # [seq_length, batch_size, feature_size]
    enc_inputs = enc_inputs.to(device)
    labels = (labels-1).to(device)   # label begins from 0
    optimizer.zero_grad()
    output = model.forward(enc_inputs)
    #########
    ##loss
    loss = criterion(output, labels.long())
    # l2 reg
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)
    loss += (modelConfig["l2_lambda"]*l2_reg)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    #########
    ## acc
    train_acc = eval_acc(data.to(device),labels,model)
    test_acc = eval_acc(test_data.to(device), test_label.to(device), model)

    return loss, train_acc, test_acc

    


def pre_train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    #####################################
    ## Dataset
    #####################################
    train_set, test_set = GetDataSet(modelConfig)
    trainloader = DataLoader(
        train_set, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

    ###########################################
    ## Model
    ###########################################
    seq_length = train_set.data.shape[1]
    feature_size = train_set.data.shape[2]
    #out_size = feature_size     # reconstruction
    out_size = len(np.unique(train_set.label))      # classification
    d_model = modelConfig["d_model"] 
    dropout = modelConfig["dropout"]

    model = TransAm(feature_size, out_size, seq_length, d_model, dropout = dropout).to(device)
    #criterion = nn.MSELoss().to(device)   
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW( model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer = optimizer, multiplier = modelConfig["multiplier"],
                                             warm_epoch = modelConfig["epoch"] // 10, after_scheduler = cosineScheduler)
    print("Model build done.")

    ###########################################
    ## log
    ###########################################
    writer = SummaryWriter(modelConfig["log_dir"])
    if os.path.exists(modelConfig["save_dir"]) == False: os.makedirs(modelConfig["save_dir"])

    #############################################
    ## Train
    #############################################
    print("Training start ......")
    test_data = torch.from_numpy(test_set.data)
    test_label = torch.from_numpy(test_set.label)
    for e in range(modelConfig["epoch"]):
        epoch_loss, epoch_train_acc, epoch_test_acc, iteration = 0.0, 0.0, 0.0, 0
        with tqdm(trainloader, dynamic_ncols=True) as tqdmDataLoader:
            for data, labels in tqdmDataLoader:
                # train a step
                loss, train_acc, test_acc = step(data, labels, test_data, test_label, 
                    model, optimizer, criterion, device, modelConfig)
                
                # compute loss
                epoch_loss += loss.item()
                epoch_train_acc += train_acc
                epoch_test_acc += test_acc
                iteration += 1
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    #"img shape: ": data.shape,
                    "train acc: ": train_acc,
                    "test acc: ": test_acc,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        #########################################
        writer.add_scalar('loss', float(epoch_loss/iteration), e)
        writer.add_scalar('train acc', float(epoch_train_acc/iteration), e)
        writer.add_scalar('test acc', float(epoch_test_acc/iteration), e)
        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]["lr"], e)
        # save model every 20 epoches
        if (e+1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(
                modelConfig["save_dir"], 'ckpt_' + str(e+1) + "_.pt"))


def kmeans_select(ckpt, dataset, model, NUM_CLUSTER, LAMBDA, device):
    with torch.no_grad():
        # extract features
        if not torch.is_tensor(dataset.data):
            dataset.data = torch.from_numpy(dataset.data).to(device)
        enc_inputs = dataset.data.permute([1,0,2])  # [seq_length, batch_size, feature_size]
        features = model.encode_out(enc_inputs)   # [seq_length, batch_size, d_model]
        features = features[0]   # [batch, d_model]  使用seq第一个向量作分类
        
        # clustering
        kmeans = KMeans(n_clusters=NUM_CLUSTER).fit(features.cpu().detach().numpy())
        
        # select centers
        distances = kmeans.transform(features.cpu().detach().numpy()) # num imgs * NUM_CLUSTER
        center_idx = np.argmin(distances, axis=0)
        centers = features[center_idx]
        features_T = torch.from_numpy(features.cpu().detach().numpy().T)

        # calculate similarity matrix
        similarities = torch.matmul(centers.cpu(), features_T) # NUM_CLUSTER * num images
        # normal to 0-1
        similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        # l2 normalize
        # similarities = F.normalize(similarities, p=2, dim=1)

        # select reliable images
        reliable_image_idx = np.unique(np.argwhere(similarities.cpu().detach().numpy() > LAMBDA)[:,1])
        # print ('ckpt %d: # reliable datapoints %d'%(ckpt, len(reliable_image_idx)))
        # sys.stdout.flush()
        dataset_select = copy.deepcopy(dataset)
        data,lable = dataset[reliable_image_idx]
        dataset_select.set_data(data, lable)

        # compute distance
        distances = distances.min(axis=1)   #(num samples, )
        distance_sum = F.normalize(torch.from_numpy(distances),p=2,dim=0).sum()

    return dataset_select, distance_sum

def fine_tune(modelConfig:Dict):
    device = torch.device(modelConfig["device"])

    #####################################
    ## Dataset
    #####################################
    dataset, test_set = GetDataSet(modelConfig)
    NUM_CLUSTER = len(np.unique(dataset.label))

    ###########################################
    ## Model
    ###########################################
    #data_size = dataset.data.shape[0]
    seq_length = dataset.data.shape[1]
    feature_size = dataset.data.shape[2]
    out_size = NUM_CLUSTER     # classification
    d_model = modelConfig["d_model"] 
    model = TransAm(feature_size, out_size, seq_length, d_model).to(device)


    # learning
    START = 1
    END = 25
    LAMBDA = 0.9
    NUM_EPOCH = 20
    BATCH_SIZE = 10
    

    #######################################
    ## Iterate
    #######################################

    for ckpt in range(START, END+1):
        # get initial model
        model_path = r"E:\Character Motion\PUL-Transformer\Fine_tune_model\1\ckpt_{}_.pt".format(ckpt-1)
        dict = torch.load(model_path)
        model.load_state_dict(dict) 
        
        # do kmeans and selecte the datapoints 
        dataset_select, distance = kmeans_select(ckpt, dataset, model, NUM_CLUSTER, LAMBDA, device)
        print("Iteration {}/{}|| Selected training data shape: {}".format(ckpt, END, dataset_select.data.shape))

        # prepare training dataset
        trainloader = DataLoader(dataset_select, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        # retrain: fine tune
        path = r"E:\Character Motion\PUL-Transformer\Fine_tune_model\1\ckpt_0_.pt".format(ckpt-1)
        dict_0 = torch.load(path)   # 在最原始的model上finetune
        model.load_state_dict(dict_0) 

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW( model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
        warmUpScheduler = GradualWarmupScheduler(optimizer = optimizer, multiplier = modelConfig["multiplier"],
                                                warm_epoch = modelConfig["epoch"] // 10, after_scheduler = cosineScheduler)
        
        ###########################################
        ## log
        ###########################################
        writer = SummaryWriter(modelConfig["finetune_log_dir"] + '/iteration_' + str(ckpt))
        if os.path.exists(modelConfig["finetune_dir"]) == False: os.makedirs(modelConfig["finetune_dir"])

        #############################################
        ## Train
        #############################################
        print("Iteration {}/{}|| Retraining start ......".format(ckpt, END))
        test_data = torch.from_numpy(test_set.data)
        test_label = torch.from_numpy(test_set.label)
        for e in range(NUM_EPOCH):
            epoch_loss, epoch_train_acc, epoch_test_acc, iteration = 0.0, 0.0, 0.0, 0
            epoch_k_loss = 0.0
            # kmeans update
            if (e+1)  % modelConfig['kmeans_epoch'] == 0:
                _, distance = kmeans_select(ckpt, dataset, model, NUM_CLUSTER, LAMBDA, device)
            with tqdm(trainloader, dynamic_ncols=True) as tqdmDataLoader:
                for data, labels in tqdmDataLoader:
                    # train a step
                    enc_inputs = data.permute([1,0,2])     # [seq_length, batch_size, feature_size]
                    enc_inputs = enc_inputs.to(device)
                    labels = (labels-1).to(device)   # label begins from 0
                    optimizer.zero_grad()
                    output = model.forward(enc_inputs)

                    ########################
                    # loss
                    loss = criterion(output, labels.long())
                    # kmeans loss
                    k_loss = distance
                    # l2 reg
                    l2_reg = torch.tensor(0.).to(device)
                    for param in model.parameters():
                        l2_reg += torch.norm(param, p=2)
                    
                    loss += (modelConfig["l2_lambda"]*l2_reg) + (modelConfig["kmeans_lambda"]*k_loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    # acc
                    train_acc = eval_acc(data.to(device),labels,model)
                    test_acc = eval_acc(test_data.to(device), test_label.to(device), model)
                    
                    # compute loss
                    epoch_loss += loss.item()
                    epoch_k_loss += k_loss
                    epoch_train_acc += train_acc
                    epoch_test_acc += test_acc
                    iteration += 1
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "iteration": ckpt,
                        "epoch": e,
                        "loss: ": loss.item(),
                        "train acc: ": train_acc,
                        "test acc: ": test_acc,
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    })
            warmUpScheduler.step()
            #########################################
            writer.add_scalar('total_loss', float(epoch_loss/iteration), e)
            writer.add_scalar('kmeans_loss', float(epoch_k_loss/iteration), e)
            writer.add_scalar('train acc', float(epoch_train_acc/iteration), e)
            writer.add_scalar('test acc', float(epoch_test_acc/iteration), e)
            writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]["lr"], e)
            
            gc.collect()
            torch.cuda.empty_cache()
        torch.save(model.state_dict(), os.path.join(
                modelConfig["finetune_dir"], 'ckpt_' + str(ckpt) + "_.pt"))
