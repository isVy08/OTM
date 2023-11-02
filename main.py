import torch
import os, sys
from config import get_data
from utils.missing import MAE, RMSE

config_id = int(sys.argv[1])
graph_type = sys.argv[2] # ER, SF
sem_type = sys.argv[3]
# action = sys.argv[4]
dataset, config = get_data(config_id, graph_type, sem_type)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = torch.from_numpy(dataset.X).double().to(device)
X_true = torch.from_numpy(dataset.X_true).double().to(device)
mask = torch.isnan(data).to(torch.float32)
N, D = data.shape

from model import MissModel, DagmaNonlinear
hidden_dims = [D, D, 1]
miss_model = MissModel(data, mask, hidden_dims, device)
miss_model.to(device)
model = DagmaNonlinear(miss_model)

W_est = model.fit(lambda1=0.02, lambda2=0.005) # , warm_iter=300, max_iter=30
from utils.eval import evaluate
raw_result = evaluate(dataset.B_bin, W_est, threshold = 0.3)
