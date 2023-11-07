import torch, ot
import torch.nn as nn
import numpy as np
from utils.arch import MLP
from utils.missing import nanmean
from tqdm import tqdm
import typing, copy, random
from model import SuperImputer
from gp import SCM


        

class MissModel(nn.Module):
    def __init__(self, data, mask, hidden_dims, device, sem_type, initialized = None):
        super(MissModel, self).__init__()

        self.d = hidden_dims[0]
        self.sem_type = sem_type

        self.scm = SCM(hidden_dims, device=device, bias=True)
        self.imputer = SuperImputer(data, mask, [self.d, self.d], initialized)
        
        
    def to_adj(self):
        return self.scm.fc1_to_adj()

    def forward(self):
        '''
        x : torch.Tensor shape (N,D)
        '''
        x = self.imputer()      
        # reconstruction from the imputations
        xhat = self.scm(x) 
    
        return x, xhat
    
class CustomNonlinear:
    """
    Class that implements the DAGMA algorithm
    """
    
    def __init__(self, model: nn.Module, verbose: bool = False):
        """
        Parameters
        ----------
        model : nn.Module
            Neural net that models the structural equations.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit`. Defaults to ``False``.
        dtype : torch.dtype, optional
            float number precision, by default ``torch.double``.
        """
        self.model = model
        self.data = self.model.imputer.data
        self.mask = self.model.imputer.mask
    
    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss
    
    def mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        n, d = target.shape
        loss = 0.5 / n * torch.sum((output - target) ** 2)
        return loss
    
    def exact_ot_cost(self, X, sample_size = 10):
        '''
        x, y are transposed.
        '''

        s1 = random.sample(range(self.model.d), k = sample_size)
        remain = [i for i in range(self.model.d) if i not in s1]
        s2 = random.sample(remain, k = sample_size)

        x = X[:, s1].t()
        y = X[:, s2].t()


        fc1_weight = self.model.to_adj()
        W = torch.square(fc1_weight)
        cost_fn = torch.nn.functional.mse_loss

        dim  = x.shape[0]
        unif = torch.ones((dim,), device = x.device) / dim
        
        M = torch.zeros((dim, dim), device = x.device)
        for i in range(dim): 
            for j in range(dim):
                a, b = s1[i], s2[j]
                weight = max(1 - W[a,b], 0)
                ml = weight * cost_fn(x[i, :], y[i, :])
                ml = W[a,b] * cost_fn(x[i, :], y[i, :])
                M[i,j] = ml 
        
        loss = ot.emd2(unif, unif, M)
        return loss 

    def fit(self, lambda1,lambda2, max_iter, lr=1e-3):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for i in tqdm(range(max_iter)):
            
            X, Xhat = self.model()
            score = self.mse_loss(Xhat, X)
            h_val = self.model.scm.h_func()
            
            l1_reg = lambda1 * self.model.scm.fc1_l1_reg()
            obj = score + l1_reg + lambda2 * h_val 

            coefs = {'mlp': 0.01, 'gp': 0.001}
            obj = obj + coefs[self.model.sem_type] * self.exact_ot_cost(X)
   
            optimizer.zero_grad()
            obj.backward(retain_graph=True)
            optimizer.step()
            
        W_est = self.model.to_adj()
    
    
        return W_est
