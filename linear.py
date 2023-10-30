import torch
import numpy as np
import torch.nn as nn 
from simple_dag import SimpleDAG


class SCM(nn.Module):
    def __init__(self, D, device):
        super(SCM, self).__init__()

        self.D = D
        self.device = device
        self.I = torch.eye(self.D, device = device)
    
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.empty(D, D)))
        self.bias = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, D)))
        self.graph = SimpleDAG(self.D, device)
        
    
    def to_adj(self):
    
        if self.training:
            A = self.graph.sample()
        else: 
            A = self.graph.get_prob_mask()
        
        W = self.weight * A

        return W
        

    def h_func(self, method = 'gnn'):
        
        A = self.weight * self.graph.get_prob_mask()
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
    

    def forward(self, x):
        '''
        x : torch.Tensor shape (B,D)
        '''
    
        w = self.to_adj()
        # x = x - x.mean(0)
        f = x @ w + self.bias
        h_val = self.h_func()
        reg = self.l1_reg()
    

        return f, h_val, reg
