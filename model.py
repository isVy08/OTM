import ot
import torch, math
import torch.nn as nn
import numpy as np
from utils.arch import MLP
from utils.missing import nanmean

class SuperImputer(nn.Module): 
    def __init__(self, data, mask, imps = 0.0):
        super(SuperImputer, self).__init__()
        self.D = data.shape[1]
        self.mask = mask 
        self.data = data
        self.imps = imps
        self.mu = MLP([self.D, self.D], nn.ReLU())
        self.var = MLP([self.D, self.D], nn.ReLU())
    
    def forward(self, batch): 
        x = self.data.clone()  
        x[self.mask.bool()] = 0
        m = self.mask
       
        if batch is not None:
            x = x[batch]
            m = m[batch]
        
        imps = self.mu(x) + self.var(x) * torch.randn_like(x)
        x = imps * m + x * (1 - m)
        return x

class SimpleImputer(nn.Module): 
    def __init__(self, data, mask):
        super(SimpleImputer, self).__init__()
        imps = (torch.randn(mask.shape, device = mask.device).float() + nanmean(data, 0))[mask.bool()]
        self.imps = nn.Parameter(imps)
        self.data = data 
        self.mask = mask
    
    def forward(self, batch): 
        x = self.data.clone()  
        x[self.mask.bool()] = self.imps
        m = self.mask
       
        if batch is not None:
            x = x[batch]
            m = m[batch]
        
        return x

class MissModel(nn.Module):
    def __init__(self, data, mask, device, hidden_dims, sem_type, imp_type, criterion):
        super(MissModel, self).__init__()

        self.N, self.D = data.shape
        self.mask = mask 
        self.data = data
        
        self.sem_type = sem_type
        self.criterion = criterion
        
        if self.sem_type == 'linear':
            from linear import SCM
            self.scm = SCM(self.D, device)
        elif self.sem_type == 'gp':
            from gp import SCM
            self.scm = SCM(self.D, device, hidden_dims)
        else:
            from mlp import SCM
            self.scm = SCM(self.D, device, hidden_dims)
        
        if imp_type == 'simple': 
            self.imputer = SimpleImputer(data, mask)
        else:
            self.imputer = SuperImputer(data, mask)
        
        # self.encoder = MLP([self.D, self.D // 2])

       


    
    def to_adj(self):
        return self.scm.to_adj()

    def forward(self, batch):
        '''
        x : torch.Tensor shape (N,D)
        '''
        x = self.imputer(batch)

        # reconstruction from the imputations
        f, h_val, reg = self.scm(x) 

        # z = self.encoder(x)
        # g = self.encoder(f)
        
        loss = self.criterion.loss_fn(f, x)
        # loss = loss + self.criterion.loss_fn(g, z)
        loss = loss + 0.01 * self.criterion.ot_dist(f, x)


        if self.criterion.alpha is not None:
            loss = loss + self.criterion.alpha * h_val
        
        if self.criterion.gamma is not None:
            loss = loss + self.criterion.gamma * reg
        return loss



    
