import torch
import torch.nn as nn
import torch.nn.functional as F


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.BN = nn.BatchNorm1d(19622)
        self.atten_m = nn.Sequential(nn.Linear(19622,1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 19622),
            nn.BatchNorm1d(19622),
            nn.GELU())

        self.atten_h = nn.Sequential(nn.Linear(1553,128,bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 1553),
            nn.BatchNorm1d(1553),
            nn.GELU())
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(19622, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(19622, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(256,1)
)
        self.reset_parameters()
    #150000个参数
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.atten_m[4].weight, a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.atten_m[0].weight, a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.atten_h[0].weight, a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.atten_h[4].weight,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.linear_relu_stack[0].weight, a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_uniform_(self.linear_relu_stack[4].weight, a=0,mode='fan_in',nonlinearity='relu')


    def forward(self, x):
        xv_m,_ = torch.max(x,dim=1)
        xa_m = torch.mean(x,dim=1)
        xv_h,_ = torch.max(x,dim=2)
        xa_h = torch.mean(x,dim=2)
        x_m = self.atten_m(xv_m) + self.atten_m(xa_m)
        x_m = self.sigmoid(x_m)
        x_h = self.atten_h(xv_h) + self.atten_h(xa_h)
        x_h = self.softmax(x_h).unsqueeze(2)
        x = (x_h * x).sum(dim=1)
        x = x_m * x
        logits = self.linear_relu_stack(x)
        logits = self.sigmoid(logits)
        return logits

    def evaluate(logits,labels):
        logits[logits>0.5] = 1
        logits[logits<=0.5] = 0
        correct = torch.sum(logits == labels)
        return correct.item()