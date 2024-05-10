import torch
import sys
from config import get_data
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

from imputers import OTimputer, RRimputer
from utils.missing import pick_epsilon

config_id = int(sys.argv[1])
graph_type = sys.argv[2] # ER, SF
sem_type = sys.argv[3]
method = sys.argv[4] # imputation method

dataset, config = get_data(config_id, graph_type, sem_type)

code = f"{config['code']}-{method}"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if graph_type == 'REAL':
    dataset.X = dataset.X.astype("float32") 

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
    d_ = d - 1
    models = {}

    for i in range(d):
        models[i] = torch.nn.Linear(d_, 1)
        models[i].to(device)

    #Create the imputer
    lin_rr_imputer = RRimputer(models, eps=epsilon, lr=lr)
    X_filled, _, _ = lin_rr_imputer.fit_transform(X, verbose=True, X_true=X_true)
    X_filled = X_filled.cpu().detach().numpy()

elif method == 'mean':
    X_filled = SimpleImputer().fit_transform(dataset.X)

elif method in ('iterative', 'missforest'):

    if method == 'missforest': 
        from sklearn.ensemble import RandomForestRegressor
        estimator =  RandomForestRegressor(n_estimators=4, max_depth=10, 
                                            bootstrap=True, max_samples=0.5, 
                                            n_jobs=2, random_state=0)
        X_filled = IterativeImputer(estimator=estimator, random_state=0, max_iter=50).fit_transform(dataset.X)
    
    else:
        # linear Bayesian Ridge by default
        X_filled = IterativeImputer(random_state=0, max_iter=50).fit_transform(dataset.X)

    

print(f'Learning DAG for {config["sem_type"]} model begins ...')

if config['sem_type'] == 'linear':
    from dag_methods import Notears
    model = Notears(lambda_1_ev=0.1)
    if isinstance(X_filled, torch.Tensor):
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


saved_path = f'output/baseline_{sem_type}.txt'
from utils.eval import evaluate, write_result
if sem_type == 'neuro' or 'dream' in sem_type:
    raw_result = evaluate(dataset.B_bin, W_est, threshold = 0.3, prune=False)
else:
    raw_result = evaluate(dataset.B_bin, W_est, threshold = 0.3)

write_result(raw_result, code, saved_path)
