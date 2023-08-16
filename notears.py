from config import get_config
from synthetic import SyntheticDataset
from torch.utils.data import DataLoader
from utils.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
import os, sys

set_seed(8)


action = sys.argv[1]
config_id = 6
device_id = 0


config = get_config(config_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', device_id)

print('Loading data ...')
dataset = SyntheticDataset(n = config['num_obs'], d = config['num_vars'], 
                           config_id = config_id,
                           graph_type = config['graph_type'], 
                           degree = config['degree'], 
                           noise_type = config['noise_type'],
                           miss_type = config['miss_type'], 
                           miss_percent = config['miss_percent'], 
                           sem_type = 'mlp' if config['sem_type'] != 'linear' else config['sem_type'],
                           equal_variances = config['ev']
                           )

X_true = torch.from_numpy(dataset.X_true).to(device)
X = torch.from_numpy(dataset.X).to(device)



N, D = X.shape
batchsize = 150

class Model(nn.Module):
    def __init__(self, D):
        super(Model, self).__init__()

        # Weighted adjacency matrix
        self.D = D
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(2 * self.D * self.D)))
        
    def _adj(self):

        W = torch.nn.functional.relu(self.weight)
        W = (W[: self.D * self.D] - W[self.D * self.D :]).reshape(self.D, self.D)
        W.fill_diagonal_(0)
        return W

    def forward(self, X):
        '''
        X : torch.Tensor shape (N,D)
        '''
        W = self._adj()
        F = X @ W
        return F
    
model_path = dataset.data_path.replace('pickle', 'pt').replace('./dataset/', f'./models/')

model = Model(D)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.90)

if os.path.isfile(model_path):
    print('Loading model ...')
    from utils.io import load_model
    prev_loss = load_model(model, optimizer, scheduler, model_path, device)

else:
    prev_loss = 10


model.to(device)

def train_epoch(X, model, optimizer, loader, device, lda):
    
    n, d = X.shape
    total = 0
    loss_fn = torch.nn.MSELoss()    
 
    for idx in loader:   
        
        loss = 0 
        x = X[idx, ].to(device)
        f = model(x)
        loss = loss_fn(f, x)
            
        W = model._adj()
        e = torch.exp(W * W)
        h = torch.trace(e) - d
        reg = 0.5 * h * h + h + 0.1 * W.sum()
        loss = loss + lda * reg
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total += loss

    return total / len(loader)

if action == 'train':

    model.train()

    indices = list(range(N))
    loader = DataLoader(indices, batch_size=batchsize, shuffle=True)
    niter = 5000

    print('Training begins:')
    for i in range(niter):

        loss = train_epoch(X, model, optimizer, loader, device, lda = 0.001)
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Nan or inf loss")
            break
        loss = loss.item()
        print(f'Iteration {i}:\t Loss: {loss:.4f}')
        
        if loss < prev_loss:
            torch.save({'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'prev_loss': loss, 
                        }, model_path)
        

else:
    from utils.utils import evaluate
    evaluate(X, dataset.B_bin, model, None, 0.3, False, X_true = X_true, prune = True)
 