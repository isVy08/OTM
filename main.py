from trainer import nanmean, MAE, RMSE, train_epoch
from config import get_config
from synthetic import SyntheticDataset
from torch.utils.data import DataLoader
from utils.utils import set_seed
from model import Model
import numpy as np
import torch
import os, sys

set_seed(8)

config_id = int(sys.argv[1])
action = sys.argv[2]
device_id = int(sys.argv[3])


config = get_config(config_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', device_id)

print('Loading data ...')
dataset = SyntheticDataset(n = config['num_obs'], d = config['num_vars'], 
                           graph_type = config['graph_type'], 
                           degree = config['degree'], 
                           noise_type = config['noise_type'],
                           miss_type = config['miss_type'], 
                           miss_percent = config['miss_percent'], 
                           sem_type = 'mlp' if config['sem_type'] != 'linear' else config['sem_type'],
                           equal_variances = config['ev']
                           )

from sklearn.preprocessing import scale
X = scale(dataset.X)
X_true = scale(dataset.X_true)
X = torch.from_numpy(X).to(device)
X_true = torch.from_numpy(X_true).to(device)


n, d = X.shape
batchsize = 50
noise = 0.10

model_path = dataset.data_path.replace('pickle', 'pt').replace('./dataset/', f'./models/{config_id}_')

if batchsize > n // 2:
    e = int(np.log2(n // 2))
    batchsize = 2 ** e
    print(f"Batchsize larger that half size = {len(dataset.X) // 2}. Setting batchsize to {batchsize}.")

model = Model(config['sem_type'], d)
mask = torch.isnan(X).float()
if os.path.isfile(model_path):
    print('Loading model ...')
    from utils.io import load_model
    _, imps = load_model(model, None, None, model_path, device)

else:
    imps = (noise * torch.randn(mask.shape, device = device).float() + nanmean(X, 0))[mask.bool()]


model.to(device)


if action == 'train':

    model.train()
    imps = imps.to(device)
    imps.requires_grad = True

    params = [imps] + list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    indices = list(range(n))
    loader = DataLoader(indices, batch_size=batchsize, shuffle=True)
    niter = 200

    print('Training begins:')
    for i in range(niter):
        X_filled, loss = train_epoch(X, model, optimizer, mask, imps, loader, 
                    batchsize, device, lda = 1)

        mae = MAE(X_filled, X_true, mask).item()
        rmse = RMSE(X_filled, X_true, mask).item()
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            ### Catch numerical errors/overflows (should not happen)
            print("Nan or inf loss")
            break
        loss = loss.item()
        print(f'Iteration {i}:\t Loss: {loss:.4f}\t '
                        f'Validation MAE: {mae:.4f}\t'
                        f'RMSE: {rmse:.4f}')
        
        torch.save({'model_state_dict': model.state_dict(), 
                    'prev_loss': loss, 
                    'imputes': imps}, model_path)

else:
    X_filled = X.detach().clone()
    X_filled[mask.bool()] = imps

    print(X_true[mask.bool()])
    print(imps)
    mae = MAE(X_filled, X_true, mask).item()
    rmse = RMSE(X_filled, X_true, mask).item()
    print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')

    B_est = model.linear.weight
    B_est = B_est.detach().cpu().numpy()
    
    
    from utils.utils import MetricsDAG, is_dag, postprocess
    _, B_out = postprocess(B_est, graph_thres = 0.30)

    # from utils.utils import print_edges
    # print(B_est.max(), np.abs(B_est).min(), np.abs(B_est).mean())
    # print_edges(B_out, 15)
    # print_edges(dataset.B_bin, 15)
    
    print('Is DAG?', is_dag(B_out))
    raw_result = MetricsDAG(B_out, dataset.B_bin)
    raw_result.display()

           