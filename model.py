import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, D, H):
        super(MLP, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(D, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, D),
        ) 
    
    def forward(self, X):
        return self.linear(X)


class Model(nn.Module):
    def __init__(self, sem_type, D, H = None):
        super(Model, self).__init__()

        # Weighted adjacency matrix
        self.D = D
        self.linear = nn.Linear(D, D, bias=False)

        self.sem_type = sem_type
        if sem_type != 'linear':
            if H is None: H = 10
            self.layer = MLP(D, H) 

    def forward(self, X):
        '''
        X : torch.Tensor shape (N,D)
        '''
        
        if self.sem_type != 'linear':
            X = self.layer(X)
            
        return self.linear(X)
            