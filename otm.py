import torch
import os, sys
import numpy as np
from config import get_data
from utils.missing import MAE, RMSE
from synthetic import SyntheticDataset
from torch.utils.data import DataLoader
from utils.eval import set_seed
from trainer import Criterion


config_id = int(sys.argv[1])
graph_type = sys.argv[2] # ER, SF
sem_type = sys.argv[3]
action = sys.argv[4]
dataset, config = get_data(config_id, graph_type, sem_type)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = torch.from_numpy(dataset.X).to(torch.float32).to(device)
X_true = torch.from_numpy(dataset.X_true).to(torch.float32).to(device)
mask = torch.isnan(X).to(torch.float32)
N, D = X.shape



alpha = 0.1
gamma = 0.001
batchsize = 500


# def loss_fn(x, y):
#     return 0.5 / x.shape[0] * ((x - y) ** 2).sum() 

loss_fn = torch.nn.functional.mse_loss
ground_cost = 'exact-ot'
methods = None
criterion = Criterion(alpha, gamma, ground_cost, methods, loss_fn)



from model import MissModel
hidden_dims = [D, D, D]
niter = 100
imp_type = 'simple'

model = MissModel(X, mask, device, hidden_dims, sem_type, imp_type, criterion)

model_path = f'models/{config["code"]}.pt'
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


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
        X_filled, loss = train_epoch(model, optimizer, loader)
        
                
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

    from utils.eval import evaluate
    B_est = model.to_adj() 
    evaluate(dataset.B_bin, B_est, threshold = 0.3, prune = True)
    print('==============================')
    

