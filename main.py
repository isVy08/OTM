import torch
import os, sys
from config import get_data
from utils.missing import MAE, RMSE

config_id = int(sys.argv[1])
graph_type = sys.argv[2] # ER, SF
sem_type = sys.argv[3]
# action = sys.argv[4]

dataset, config = get_data(config_id, graph_type, sem_type)
code = config["code"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = torch.from_numpy(dataset.X).to(device)
mask = torch.isnan(data)
data = data.double()
mask = mask.double()

# data = mean_imputation(dataset.X)
# data = torch.from_numpy(dataset.X).double().to(device)
X_true = torch.from_numpy(dataset.X_true).to(device)

N, D = data.shape

from model_v2 import MissModel, DagmaNonlinear
hidden_dims = [D, D, 1]
miss_model = MissModel(data, mask, hidden_dims, device, sem_type, initialized = None)
miss_model.to(device)
model = DagmaNonlinear(miss_model)

if sem_type == 'gp':
    W_est = model.fit(lambda1=0.02, lambda2=0.005, warm_iter=3000, max_iter=3000)
else:
    W_est = model.fit(lambda1=0.02, lambda2=0.005)

# W_est = model.fit(lambda1=0.02, lambda2=0.005, warm_iter=3, max_iter=3) # testing only

# =============== WRITE GRAPH ===============
from utils.eval import evaluate, write_result
raw_result = evaluate(dataset.B_bin, W_est, threshold = 0.3)
saved_path = f'output/otm_{sem_type}.txt'
write_result(raw_result, code, saved_path)


model_path = f'models/{code}.pt'
print(f'Saving model at {model_path}...')
torch.save({'model_state_dict': model.model.state_dict()}, model_path)


# =============== WRITE IMPUTATION ===============
X_filled = model.model.imputer()
mae = MAE(X_filled, X_true, mask).item()
rmse = RMSE(X_filled, X_true, mask).item()
print(mae, rmse)

file = open(f'output/otm_{sem_type}_imputation.txt', 'a+')
file.write(f'{config["code"]}\n')
file.write(f'MAE: {mae}, RMSE: {rmse}\n')
file.write('======================\n')
file.close()