import torch
import os, sys
import numpy as np
from config import get_config
from utils.missing import MAE, RMSE
from synthetic import SyntheticDataset
from torch.utils.data import DataLoader
from utils.eval import set_seed



action = sys.argv[1]
config_id = int(sys.argv[2])
device_id = int(sys.argv[3])
name = sys.argv[4]

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
                           sem_type = config['sem_type'],
                           equal_variances = config['ev']
                           )


X = torch.from_numpy(dataset.X).to(device)
X_true = torch.from_numpy(dataset.X_true).to(device)
mask = torch.isnan(X).float()
N, D = X.shape



alpha = 0.1
beta = 0.01
gamma = 0.001
from model_mcar import MissModel, Criterion
hidden_dims = [D, D, D]


batchsize = 300





if config_id in (13,14):
    niter = 3000
elif config_id in (15,16):
    niter = 5000 
else:
    niter = 10000

lr = 1e-3

loss_fn = None
ground_cost = 'exact-ot'
methods = ('got', )
use_gan = False
criterion = Criterion(alpha, beta, gamma, ground_cost, methods, loss_fn)
model = MissModel(X, mask, device, hidden_dims, config)




if name == 'default':
    model_path = dataset.data_path.replace('pickle', 'pt').replace('./dataset/', f'./models/')
else: 
    model_path = f'models/{config_id}_{name}.pt'



optimizer = torch.optim.Adam(model.parameters(), lr=lr)


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

    
    
    from trainer import train_epoch
    
    print('Training begins:')
    for i in range(niter):   
        X_filled, loss = train_epoch(model, optimizer, loader, criterion)
        
                
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
    evaluate_dag(dataset.B_bin, B_est, threshold = 0.3, prune = True)
    print('==============================')
    # evaluate_dag(dataset.B_bin, B_est, threshold = 0.3, prune = False)
    

