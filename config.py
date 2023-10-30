from synthetic import SyntheticDataset
from utils.eval import set_seed

set_seed(8)

def get_linear_config(config_id, graph_type):
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
            "sem_type": 'linear',
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
    
    config['code'] = f'Linear-{graph_type}{config_id}'
    
    return config


def get_data(config_id, graph_type, sem_type):
    if sem_type == 'linear':
        config = get_linear_config(config_id, graph_type)
    
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


def get_config(config_id):
    configs = {

        1: {
            'num_obs': 100,
            'num_vars': 30,
            'graph_type': 'ER',
            'degree': 2, 
            'noise_type': 'gaussian',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'linear',
            "ev": True
        }, 

        9: {
            'num_obs': 10000,
            'num_vars': 100,
            'graph_type': 'ER',
            'degree': 2, 
            'noise_type': 'gaussian',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'linear',
            "ev": True
        },

        2: {
            'num_obs': 100,
            'num_vars': 30,
            'graph_type': 'ER',
            'degree': 2, 
            'noise_type': 'gaussian',
            'miss_type': 'mcar',
            'miss_percent': 0.5,
            "sem_type": 'linear',
            "ev": True
        }, 

        3: {
            'num_obs': 1000,
            'num_vars': 50,
            'graph_type': 'ER',
            'degree': 2, 
            'noise_type': 'gaussian',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'linear',
            "ev": True
        }, 

        4: {
            'num_obs': 1000,
            'num_vars': 50,
            'graph_type': 'ER',
            'degree': 2, 
            'noise_type': 'gaussian',
            'miss_type': 'mnar',
            'miss_percent': 0.1,
            "sem_type": 'linear',
            "ev": True
        }, 
        ####################
         5: {
            'num_obs': 100,
            'num_vars': 30,
            'graph_type': 'SF',
            'degree': 2, 
            'noise_type': 'uniform',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'linear',
            "ev": False
        }, 


        10: {
            'num_obs': 10000,
            'num_vars': 100,
            'graph_type': 'SF',
            'degree': 2, 
            'noise_type': 'uniform',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'linear',
            "ev": False
        }, 

        6: {
            'num_obs': 100,
            'num_vars': 30,
            'graph_type': 'SF',
            'degree': 2, 
            'noise_type': 'uniform',
            'miss_type': 'mcar',
            'miss_percent': 0.5,
            "sem_type": 'linear',
            "ev": False
        }, 

        7: {
            'num_obs': 1000,
            'num_vars': 50,
            'graph_type': 'SF',
            'degree': 2, 
            'noise_type': 'uniform',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'linear',
            "ev": False
        }, 

        8: {
            'num_obs': 1000,
            'num_vars': 50,
            'graph_type': 'SF',
            'degree': 2, 
            'noise_type': 'uniform',
            'miss_type': 'mnar',
            'miss_percent': 0.1,
            "sem_type": 'linear',
            "ev": False
        }, 

        ####################
        11: {
            'num_obs': 100,
            'num_vars': 30,
            'graph_type': 'SF',
            'degree': 2, 
            'noise_type': 'uniform',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'mlp',
            "ev": False
        }, 

        12: {
            'num_obs': 1000,
            'num_vars': 50,
            'graph_type': 'ER',
            'degree': 2, 
            'noise_type': 'gaussian',
            'miss_type': 'mcar',
            'miss_percent': 0.5,
            "sem_type": 'mlp',
            "ev": True
        }, 

        13: {
            'num_obs': 100,
            'num_vars': 30,
            'graph_type': 'SF',
            'degree': 2, 
            'noise_type': 'gumbel',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'gp',
            "ev": False
        }, 

        14: {
            'num_obs': 1000,
            'num_vars': 50,
            'graph_type': 'ER',
            'degree': 2, 
            'noise_type': 'laplace',
            'miss_type': 'mcar',
            'miss_percent': 0.5,
            "sem_type": 'gp',
            "ev": True
        }, 

         15: {
            'num_obs': 1000,
            'num_vars': 50,
            'graph_type': 'SF',
            'degree': 2, 
            'noise_type': 'gaussian',
            'miss_type': 'mnar',
            'miss_percent': 0.5,
            "sem_type": 'mlp',
            "ev": True
        }, 

        16: {
            'num_obs': 1000,
            'num_vars': 50,
            'graph_type': 'SF',
            'degree': 2, 
            'noise_type': 'laplace',
            'miss_type': 'mnar',
            'miss_percent': 0.5,
            "sem_type": 'gp',
            "ev": True
        }, 

        

        
        

    }

    return configs[config_id]
