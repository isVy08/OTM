import torch
import os, sys
import numpy as np
from config import get_data
from utils.missing import MAE, RMSE
from synthetic import SyntheticDataset
from torch.utils.data import DataLoader
from utils.eval import set_seed
from model_mcar import MissModel, Criterion

config_id = int(sys.argv[1])
graph_type = sys.argv[2] # ER, SF
sem_type = sys.argv[3]
action = sys.argv[4]
dataset, config = get_data(config_id, graph_type, sem_type)
device = torch.device('mps')


X = torch.from_numpy(dataset.X).to(torch.float32).to(device)
X_true = torch.from_numpy(dataset.X_true).to(torch.float32).to(device)
mask = torch.isnan(X).to(torch.float32)
N, D = X.shape



alpha = 0.1
beta = 0.01
gamma = 0.001

hidden_dims = [D, D, D]


batchsize = 300
niter = 1000

def loss_fn(x, y):
    return 0.5 / x.shape[0] * ((x - y) ** 2).sum() 

loss_fn = torch.nn.functional.mse_loss
ground_cost = 'exact-ot'
methods = ('got', )
criterion = Criterion(alpha, beta, gamma, ground_cost, methods, loss_fn)
model = MissModel(X, mask, device, hidden_dims, config)

model_path = f'models/{config["code"]}.pt'

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-2, max_iter = 20)


if os.path.isfile(model_path):
    print(f'Loading model from {model_path} ...')
    from utils.io import load_model
    prev_loss = load_model(model, optimizer, None, model_path, device)
    

else:
    prev_loss = 30


model.to(device)

if action == 'train':

    model.train()

    indices = list(range(X.shape[0]))
    loader = DataLoader(indices, batch_size=batchsize, shuffle=True)

    from trainer import train_epoch, train_lbfgs
    
    print('Training begins:')
    for i in range(niter):   
        X_filled, loss = train_lbfgs(model, optimizer, loader, criterion)
        
                
        mae = MAE(X_filled, X_true, mask).item()
        rmse = RMSE(X_filled, X_true, mask).item()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"{loss.item()} loss")
            break

        loss = loss.item()
        if i % 10 == 0:
            print(f'Iteration {i}:\t Loss: {loss:.4f}\t '
                        f'Validation MAE: {mae:.4f}\t'
                        f'RMSE: {rmse:.4f}')
        
        if loss < prev_loss:
            print('Saving model ...')
            torch.save({'model_state_dict': model.state_dict(), 
                                'optimizer_state_dict': optimizer.state_dict(),
                                'prev_loss': loss, 
                                }, model_path)
            prev_loss = loss

else:
    model.eval()
    
    X_filled = model.impute(None)
    mae = MAE(X_filled, X_true, mask).item()
    rmse = RMSE(X_filled, X_true, mask).item()
    print(f'Validation MAE: {mae:.4f}\t'
        f'RMSE: {rmse:.4f}')

    from utils.eval import evaluate_dag
    B_est = model.to_adj() 
    evaluate(dataset.B_bin, B_est, threshold = 0.3, prune = True)
    print('==============================')
    

