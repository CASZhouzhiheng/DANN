import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import dgl
# from model import GCN
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score,precision_recall_curve, auc
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from DANN import DANN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




embedding_list = []
for i in range(737):
    file_path = '/home/anxy/zzh/gene/double_pathway/personembedding/' + 'gene_embedding' + str(i) + '.pt'
    data = torch.load(file_path)
    embedding_list.append(data.unsqueeze(0).to(torch.float32))
embedding_list = torch.cat(embedding_list,dim=0)
for i in range(1553):
    means = torch.mean(embedding_list[:,i,:],dim=0)
    stds = torch.std(embedding_list[:,i,:],dim=0)
    embedding_list[:,i,:] = (embedding_list[:,i,:] - means)/stds

label_path = '/home/anxy/zzh/gene/data_label.txt'
label_file = open(label_path, 'r', encoding='utf-8')
label_row = label_file.readlines()
label_list = []
for lines in label_row:
    lines = lines.rstrip()  # 删去行末的换行符
    lines = int(lines)
    label_list.append(lines)

from torch.utils import data
from torch.utils.data import Dataset
class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return len(self.data_tensor)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(embedding_list, label_list)):
    label = torch.tensor(label_list,dtype=torch.float32).to(device)

    train_data = embedding_list[train_index]
    test_data = embedding_list[test_index]
    train_label = label[train_index]
    test_label = label[test_index]

    train_dataset = TensorDataset(train_data, train_label)
    test_dataset = TensorDataset(test_data,test_label)
    train_loader = data.DataLoader(dataset=train_dataset,batch_size=16, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset,batch_size=16, shuffle=True)


    epoch_num = 30
    model = DANN().to(device) 
    loss_func = torch.nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)

    epoches=[]
    lossvalue=[]
    testlossvalue=[]
    train_accuracy = []
    test_accuracy = []
    best_acc = 0
    for epoch in range(epoch_num):
        for batched_feature ,batch_label in train_loader:
            logits = model(batched_feature.to(device)).to(device) 
            loss = loss_func(logits, batch_label.unsqueeze(1)).to(device)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            total_trainloss = 0
            total_acc = 0
            for batched_feature ,batch_label in train_loader:
                logits = model(batched_feature.to(device)).to(device) 
                acc = DNN.evaluate(logits, batch_label.unsqueeze(1))
                trainloss = loss_func(logits,batch_label.unsqueeze(1)).to(device)
                total_trainloss = total_trainloss + trainloss.detach().cpu().numpy()
                total_acc = total_acc + acc
            trainloss = total_trainloss / len(train_loader)
            acc = total_acc / len(train_data)

            print("Epoch {:05d} | Train Loss {:.16f} | Train Acc {:.16f}".format(epoch, trainloss.item(), acc))
            lossvalue.append(trainloss) 
            total_testloss = 0
            total_acc = 0
            y_true = []
            y_pred = []
            y_prob = []
            for batched_feature ,batch_label in test_loader:
                logits = model(batched_feature.to(device)).to(device)
                y_prob.append(logits.squeeze().clone())
                y_true.append(batch_label)
                test_acc = DNN.evaluate(logits, batch_label.unsqueeze(1))
                y_pred.append(logits.squeeze())
                testloss = loss_func(logits,batch_label.unsqueeze(1)).to(device)
                total_testloss = total_testloss + testloss.detach().cpu().numpy()
                total_acc = total_acc + test_acc
            testloss = total_testloss / len(test_loader)
            test_acc = total_acc / len(test_data)
            y_true = torch.cat(y_true).cpu().numpy()
            y_pred = torch.cat(y_pred).cpu().numpy()

            AUC = roc_auc_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = recall_score(y_true, y_pred)
            specificity = tn / (tn + fp)
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            aupr = auc(recall, precision)
            if test_acc > best_acc:
                best_acc = test_acc
            print("Epoch {:05d} | Test Loss {:.16f} | Test Acc {:.16f}|AUC {:.16f}|F1 {:.16f}|SEN {:.16f}|SPE {:.16f}|precision {:.16f} ".format(epoch, testloss.item(), test_acc,AUC,f1,sensitivity, specificity,precision_score_value))
            
            epoches.append(epoch)
            train_accuracy.append(acc) 
            test_accuracy.append(test_acc)
        
    print("DANN最大测试准确率：",best_acc)