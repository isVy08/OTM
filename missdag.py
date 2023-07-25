from synthetic import SyntheticDataset
from config import get_config
import sys
from dag_methods import Notears, Notears_ICA_MCEM, Notears_ICA, Notears_MLP_MCEM, Notears_MLP_MCEM_INIT
from miss_methods import miss_dag_gaussian, miss_dag_nongaussian, miss_dag_nonlinear

config_id = int(sys.argv[1])
print('Loading data ...')
config = get_config(config_id)
dataset = SyntheticDataset(n = config['num_obs'], d = config['num_vars'], 
                           graph_type = config['graph_type'], 
                           degree = config['degree'], 
                           noise_type = config['noise_type'],
                           miss_type = config['miss_type'], 
                           miss_percent = config['miss_percent'], 
                           sem_type = 'mlp' if config['sem_type'] == 'logistic' else config['sem_type'],
                           equal_variances = config['ev']
                           )

print('Running EM ...')
import time 
start = time.time()
em_iter = 30
if config_id in (1,2):
    dag_method = Notears(lambda_1_ev=0.1)
    B_est, _, _ = miss_dag_gaussian(dataset.X, dataset.mask, dag_method, em_iter, config['ev'])
elif config_id == 3:
    dag_method = Notears_ICA()
    dag_init_method = Notears_ICA()
    dag_method = Notears_ICA_MCEM()
    B_est, _, _ = miss_dag_nongaussian(dataset.X, dag_init_method,
                                                    dag_method, em_iter, B_true=dataset.B_bin)
end = time.time()
print(f'Training time: {end - start}')

from utils.utils import MetricsDAG, is_dag, postprocess
_, B_out = postprocess(B_est, graph_thres = 0.30)
print('Is DAG?', is_dag(B_out))
raw_result = MetricsDAG(B_out, dataset.B_bin)
raw_result.display()
