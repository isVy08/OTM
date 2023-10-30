import ot
import torch, math
import torch.nn as nn
import numpy as np
from utils.missing import nanmean
from geomloss import SamplesLoss
from utils.arch import MLP


class Criterion:
    def __init__(self, alpha, beta, gamma, ground_cost, methods, loss_fn):
        # super(Criterion, self).__init__()
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma
        self.methods = methods

        self.loss_fn = loss_fn
        self.ground_cost = ground_cost 

        if ground_cost == 'sinkhorn':
            self.ot_dist = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=.9, backend="tensorized")
        else: 
            self.ot_dist = self.exact_ot_cost

            
    def exact_ot_cost(self, x, y, cost_fn = 'mse'):
        batchsize, _  = x.shape
        unif = torch.ones((batchsize,), device = x.device) / batchsize
        if cost_fn != 'mse':
            M = torch.zeros((batchsize, batchsize), device = x.device)
            for i in range(batchsize): 
                for j in range(batchsize):
                    ml = cost_fn(x[i:i+1, ], y[j:j+1, ])
                    M[i,j] = ml 
        else:
            # M = torch.cdist(x, y, p=2)
            M = ot.dist(x, y, metric='euclidean')
        
        loss = ot.emd2(unif, unif, M)
        return loss
        



class MissModel(nn.Module):
    def __init__(self, data, mask, device, hidden_dims, config):
        super(MissModel, self).__init__()

        self.N, self.D = data.shape
        self.mask = mask 
        self.data = data
        
        self.sem_type = config['sem_type']
        
        if self.sem_type == 'linear':
            from linear import SCM
            self.scm = SCM(self.D, device)
        elif self.sem_type == 'gp':
            from gp import SCM
            self.scm = SCM(self.D, device, hidden_dims)
        else:
            from nonlinear import SCM
            self.scm = SCM(self.D, device, hidden_dims)
        
        self.mu = MLP([self.D, self.D], nn.ReLU())
        self.var = MLP([self.D, self.D], nn.ReLU())

        # imps = (torch.randn(mask.shape, device = mask.device).float() + nanmean(data, 0))[mask.bool()]
        # self.imps = nn.Parameter(imps)


    
    def to_adj(self):
        return self.scm.to_adj()
        
    def impute(self, batch): 
        x = self.data.clone()  
        x[self.mask.bool()] = 0
        m = self.mask
       
        if batch is not None:
            x = x[batch]
            m = m[batch]
        
        imps = self.mu(x) + self.var(x) * torch.randn_like(x)
        x = imps * m + x * (1 - m)
        return x

    def forward(self, batch):
        '''
        x : torch.Tensor shape (N,D)
        '''
        x = self.impute(batch)

        # reconstruction from the imputations
        f, h_val, reg = self.scm(x) 
                
        return x, f, h_val, reg


    
