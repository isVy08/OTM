import torch
import torch.nn as nn
import sys, os
import numpy as np
from config import get_data
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from utils.missing import MAE, RMSE

from imputers import OTimputer, RRimputer
from utils.missing import pick_epsilon

config_id = int(sys.argv[1])
graph_type = sys.argv[2] # ER, SF
sem_type = sys.argv[3]
method = sys.argv[4]
dataset, config = get_data(config_id, graph_type, sem_type)

code = f"{config['code']}-{method}"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps')


X = torch.from_numpy(dataset.X).to(device)
X_true = torch.from_numpy(dataset.X_true).to(device)

mask = torch.isnan(X).float()



print('Imputing missing values begins ...')
n, d = X.shape
batchsize = 30
lr = 1e-2
epsilon = pick_epsilon(X)

if method == 'sk':
     
    sk_imputer = OTimputer(eps=epsilon, batchsize=batchsize, lr=lr, niter=10000)
    X_filled, _, _ = sk_imputer.fit_transform(X, verbose=True, report_interval=10, X_true=X_true)

elif method == 'lin-rr':

    #Create the imputation models
    output_path = f'output/linrr/{config["code"]}.npy'
    if not os.path.isfile(output_path):
        d_ = d - 1
        models = {}

        for i in range(d):
            models[i] = torch.nn.Linear(d_, 1)
            models[i].to(device)

        #Create the imputer
        lin_rr_imputer = RRimputer(models, eps=epsilon, lr=lr)
        X_filled, _, _ = lin_rr_imputer.fit_transform(X, verbose=True, X_true=X_true)
        X_filled = X_filled.cpu().detach().numpy()
        np.save(output_path, X_filled)
    else:
        X_filled = np.load(output_path)
    

elif method == 'mlp-rr':
    # Create the imputation models
    d_ = d - 1
    models = {}

    for i in range(d):
        models[i] = nn.Sequential(nn.Linear(d_, 2 * d_),
                                nn.ReLU(),
                                nn.Linear(2 * d_, d_),
                                nn.ReLU(),
                                nn.Linear(d_, 1))
        models[i].to(device) 

    #Create the imputer
    mlp_rr_imputer = RRimputer(models, eps=epsilon, lr=lr)
    X_filled, _, _ = mlp_rr_imputer.fit_transform(X, verbose=True, X_true=X_true)

elif method == 'mean':
    X_filled = SimpleImputer().fit_transform(dataset.X)

elif method == 'iterative':
    X_filled = IterativeImputer(random_state=0, max_iter=50).fit_transform(dataset.X)
    

try:
    mae = MAE(X_filled, X_true, mask).item()
    rmse = RMSE(X_filled, X_true, mask).item()
except TypeError:
    X_filled = torch.from_numpy(X_filled).to(device)
    mae = MAE(X_filled, X_true, mask).item()
    rmse = RMSE(X_filled, X_true, mask).item()


# =============== WRITE IMPUTATION ===============
file = open('output/baseline_imputation.txt', 'a+')
file.write(f'{code}\n')
file.write(f'MAE: {mae}, RMSE: {rmse}\n')
file.write('======================\n')
file.close()

print(f'Learning DAG for {config["sem_type"]} model begins ...')


if config['sem_type'] == 'linear':
    from dag_methods import Notears
    model = Notears(lambda_1_ev=0.1)
    X_filled = X_filled.cpu().detach().numpy()
    W_est = model.fit(X_filled)


else:
    from dagma import DagmaMLP, DagmaNonlinear
    
    eq_model = DagmaMLP(dims=[d, d, 1], device=device, bias=True)
    eq_model.to(device)
    model = DagmaNonlinear(eq_model)

    if not isinstance(X_filled, torch.Tensor):
        X_filled = torch.from_numpy(X_filled).to(device)

    W_est = model.fit(X_filled, lambda1=0.02, lambda2=0.005)

# =============== WRITE GRAPH ===============

if config['sem_type'] == 'linear':
    saved_path = 'output/baseline_linear.txt'
else:
    saved_path = 'output/baseline_nonlinear.txt'

from utils.eval import evaluate, write_result
raw_result = evaluate(dataset.B_bin, W_est, threshold = 0.3)

write_result(raw_result, code, saved_path)
