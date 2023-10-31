
import torch
import torch.nn as nn
import numpy as np
from utils.sampler import Sample_Bernoulli


class SimpleDAG(nn.Module):
    def __init__(self, D, device, tau = 1.2):
        super(SimpleDAG, self).__init__()

        
        self.D = D
        self.device = device
        self.I = torch.eye(self.D, device = device)
        ones = torch.ones(self.D, self.D, device = self.device)
        self.U = torch.triu(ones, diagonal = 1)
        self.L = torch.tril(ones, diagonal = -1)                 
        
        self.edge = nn.Parameter(nn.init.ones_(torch.empty(D, D)))
        self.sampler = Sample_Bernoulli(tau)
    
    def sample(self):
        
        E = torch.sigmoid(self.edge)
        E = self.sampler(E)
        A = E * self.U + (1 - E).t() * self.L
        return A

    def get_prob_mask(self):
        
        E = torch.sigmoid(self.edge) 
        A = E * self.U + (1 - E).t() * self.L
    
        return A
    
    
