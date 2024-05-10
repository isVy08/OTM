import torch
import sys
from config import get_data

config_id = int(sys.argv[1])
graph_type = sys.argv[2]
sem_type = sys.argv[3]



dataset, config = get_data(config_id, graph_type, sem_type)
code = config["code"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = torch.from_numpy(dataset.X).to(device)
mask = torch.isnan(data)


X_true = torch.from_numpy(dataset.X_true).to(device)

N, D = data.shape
hidden_dims = [D, D, 1]


from model import MissModel, DagmaNonlinear
data = data.double()
mask = mask.double()
miss_model = MissModel(data, mask, hidden_dims, device, sem_type, initialized = None)
miss_model.to(device)
model = DagmaNonlinear(miss_model)
if sem_type == 'sachs':
    W_est = model.fit(lambda1=0.02, lambda2=0.005, warm_iter=5e2, max_iter=8e2)
else: 
    W_est = model.fit(lambda1=0.02, lambda2=0.005, warm_iter=5e3, max_iter=8e3)

# =============== WRITE GRAPH ===============
from utils.eval import evaluate, write_result

if sem_type == 'neuro' or 'dream' in sem_type:
    raw_result = evaluate(dataset.B_bin, W_est, threshold = 0.3, prune=False)
else:
    raw_result = evaluate(dataset.B_bin, W_est, threshold = 0.3)

saved_path = f'output/otm_{sem_type}.txt'
write_result(raw_result, code, saved_path)
