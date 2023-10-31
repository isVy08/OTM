from synthetic import SyntheticDataset
from utils.eval import set_seed

set_seed(8)

def get_config(config_id, graph_type, sem_type):
    '''
    graph type: ER or SF
    '''

    assert config_id < 10, 'Input config id from 1 to 9'
    assert graph_type in ('ER', 'SF'), 'ER or SF graph only'
    config = {
            'num_obs': 1000,
            'num_vars': 50,
            'graph_type': graph_type,
            'degree': 2, 
            'noise_type': 'gaussian',
            # 'miss_type': 'mcar',
            # 'miss_percent': 0.1,
            "sem_type": sem_type,
            "ev": False
        }

    if config_id in (1,2,3):
        config['miss_type'] = 'mcar'
    elif config_id in (4,5,6):
        config['miss_type'] = 'mar'
    elif config_id in (7,8,9):
        config['miss_type'] = 'mnar'
    
    if config_id in (1,4,7):
        config['miss_percent'] = 0.1
    elif config_id in (2,5,8):
        config['miss_percent'] = 0.3
    if config_id in (3,6,9):
        config['miss_percent'] = 0.5
    
    if sem_type == 'linear': 
        sem_type = 'Linear'
    else: 
        sem_type = sem_type.upper()
    
    config['code'] = f'{sem_type}-{graph_type}{config_id}'
    
    return config


def get_data(config_id, graph_type, sem_type):
    config = get_config(config_id, graph_type, sem_type)
    
    dataset = SyntheticDataset(n = config['num_obs'], d = config['num_vars'], 
                           config_code = config['code'],
                           graph_type = config['graph_type'], 
                           degree = config['degree'], 
                           noise_type = config['noise_type'],
                           miss_type = config['miss_type'], 
                           miss_percent = config['miss_percent'], 
                           sem_type = config['sem_type'],
                           equal_variances = config['ev']
                           )

    return dataset, config


for config_id in range(1, 10):
    for graph_type in ('ER', 'SF'):
        get_data(config_id, graph_type, 'gp')