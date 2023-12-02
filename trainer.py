import torch
import ot
from geomloss import SamplesLoss

class Criterion:
    def __init__(self, alpha, gamma, ground_cost, methods, loss_fn):
        self.alpha = alpha 
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
    
def free_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def train_epoch(model, optimizer, loader):
    
    model.train()
   
    losses = 0
    for batch in loader:   

        loss = model(batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss

    model.eval()
    with torch.no_grad():
        X_filled = model.imputer(None)
            

    return X_filled, losses / len(loader)


