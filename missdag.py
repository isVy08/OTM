import sys
import numpy as np
from config import get_data
from dag_methods import Notears, Notears_ICA_MCEM, Notears_ICA, Notears_MLP_MCEM, Notears_MLP_MCEM_INIT
from miss_methods import miss_dag_gaussian, miss_dag_nongaussian, miss_dag_nonlinear


config_id = int(sys.argv[1])
print('Running', config_id)
graph_type = sys.argv[2] # ER, SF
sem_type = sys.argv[3]

dataset, config = get_data(config_id, graph_type, sem_type)


em_iter = 30


if config['sem_type'] == 'linear' and config['noise_type'] == 'gaussian':
    print('Running linear Gaussian model ...')
    dag_method = Notears(lambda_1_ev=0.1)
    B_est, _, _ = miss_dag_gaussian(dataset.X, dataset.mask, dag_method, em_iter, config['ev'])

elif config['sem_type'] != 'linear':
    print('Running non-linear model ...')
    dag_init_method = Notears_MLP_MCEM_INIT()
    dag_method = Notears_MLP_MCEM()
    B_est, _, _ = miss_dag_nonlinear(dataset.X, dag_init_method,
                                                    dag_method, em_iter, config['ev'])

elif config['sem_type'] == 'linear' and config['noise_type'] != 'gaussian':
    print('Running linear non-Gaussian model ...')
    dag_init_method = Notears_ICA()
    dag_method = Notears_ICA_MCEM()
    B_est, _, _ = miss_dag_nongaussian(dataset.X, dag_init_method,
                                                    dag_method, em_iter, B_true=dataset.B_bin)


from utils.eval import evaluate, write_result
saved_path = f'output/issdag_{sem_type}.txt'
if sem_type == 'neuro' or 'dream' in sem_type:
    raw_result = evaluate(dataset.B_bin, B_est, threshold = 0.3, prune=False)
else:
    raw_result = evaluate(dataset.B_bin, B_est, threshold = 0.3)
write_result(raw_result, config['code'], saved_path)
