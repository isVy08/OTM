import torch
import numpy as np
import torch.nn as nn
from utils.arch import linear_sequential
from simple_dag import SimpleDAG


class SCM(nn.Module):
    def __init__(self, hidden_dims, device, bias):
        super(SCM, self).__init__()

        self.d = hidden_dims[0]
    
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.d, self.d)))
        if bias:
            self.bias = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.d)))
        else: 
            self.bias = 0
        self.graph = SimpleDAG(self.d, device)
        self.I = torch.eye(self.d, device = device)

        layers = linear_sequential(self.d, hidden_dims[:-1], hidden_dims[-1], nn.ReLU(), 1.0, 0.0)
        self.layer = nn.Sequential(*layers)

    def fc1_to_adj(self):
        

        if self.training:
            A = self.graph.sample()
        else: 
            A = self.graph.get_prob_mask()
            
        W = self.weight * A
        return W
        

    def h_func(self, method = 'dag-gnn'):
        
        A = self.fc1_to_adj()
        A = torch.square(A)
        if method == 'dagma':
            s = 1.0
            h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
        elif method == 'notears':
            h = torch.trace(torch.matrix_exp(A)) - self.d
        else:
            M = self.I + A / self.d  # (Yu et al. 2019)
            E = torch.matrix_power(M, self.d - 1)
            h = (E.t() * M).sum() - self.d

        return h

    def fc1_l1_reg(self):
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
        w = self.fc1_to_adj()
        x = x @ w + self.bias
        e = self.__expand__(x)        
        f = self.layer(e).sum(dim = 2)
        return f


    
