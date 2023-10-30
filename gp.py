import torch
import numpy as np
import torch.nn as nn
from utils.arch import linear_sequential
from simple_dag import SimpleDAG


class SCM(nn.Module):
    def __init__(self, D, device, hidden_dims):
        super(SCM, self).__init__()

        self.D = D
    
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.empty(D, D)))
        self.bias = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, D)))
        self.graph = SimpleDAG(self.D, device)
        
        
        self.I = torch.eye(self.D, device = device)

        self.layer = linear_sequential(self.D, hidden_dims[:-1], hidden_dims[-1], nn.ReLU(), 1.0, 0.2)
    
    def split_weight(self, W):
        pW = torch.max(W, torch.zeros_like(W))
        nW = torch.max(-W, torch.zeros_like(W))
        return pW, nW

    def to_adj(self):

        if self.training:
            A = self.graph.sample()
        else: 
            A = self.graph.get_prob_mask()
            
        W = self.weight * A
        return W
        

    def h_func(self, method = 'gnn'):
        
        A = self.to_adj()
        A = torch.square(A)
        if method == 'dagma':
            s = 1.0
            h = -torch.slogdet(s * self.I - A)[1] + self.D * np.log(s)
        elif method == 'notears':
            h = torch.trace(torch.matrix_exp(A)) - self.D
        else:
            M = self.I + A / self.D  # (Yu et al. 2019)
            E = torch.matrix_power(M, self.D - 1)
            h = (E.t() * M).sum() - self.D

        return h

    def l1_reg(self):
        W = self.weight * self.graph.get_prob_mask()
        reg = torch.abs(W).sum()
        return reg
    
    def __expand__(self, x): 
        '''
        x : [N, D]
        returns [N, D, D + 1]
        '''
        x = x.unsqueeze(-1)
        I = self.I.unsqueeze(0)
        I = torch.repeat_interleave(I, repeats=x.size(0), dim=0)
        x = x * I 
        return x 

    def forward(self, x):
        '''
        x : torch.Tensor shape (B,D)
        '''
        w = self.to_adj()
        x = x @ w + self.bias
        e = self.__expand__(x)
        f = self.layer(e).sum(dim = 2)

        h_val = self.h_func()
        reg = self.l1_reg()
        return f, h_val, reg


    

