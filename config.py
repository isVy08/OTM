from synthetic import SyntheticDataset
from real import RealDataset
from utils.eval import set_seed


def get_config(config_id, graph_type, sem_type):
    '''
    graph type: ER or SF
    '''
    assert graph_type in ('ER', 'SF', 'REAL'), 'ER or SF graph only'
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

    
    if config_id > 20: 
        config['miss_type'] = 'mcar'
        config['miss_percent'] = 0.1
    
    if config_id == 21:
        config['noise_type'] = 'exponential'
    elif config_id == 22: 
        config['noise_type'] = 'laplace'
    elif config_id == 23: 
        config['noise_type'] = 'gumbel'
    elif config_id == 24: 
        config['degree'] = 4
    elif config_id == 25: 
        config['degree'] = 6
    elif config_id == 26: 
        config['degree'] = 8
    elif config_id == 27: 
        config['num_vars'] = 20
    elif config_id == 28: 
        config['num_vars'] = 30
    elif config_id == 29: 
        config['num_vars'] = 40
    elif config_id == 30: 
        config['num_vars'] = 50
    elif config_id == 31: 
        config['num_vars'] = 100
    elif config_id == 32: 
        config['num_vars'] = 200
    
    
    if sem_type == 'linear': 
        sem_type = 'Linear'
    else: 
        sem_type = sem_type.upper()
    
    config['code'] = f'{sem_type}-{graph_type}{config_id}'
    
    return config

def get_real_config(config_id, graph_type, sem_type):
    '''
    graph type: ER or SF
    '''
    assert graph_type == 'REAL'
    config = {
            'num_obs': 1000,
            'num_vars': 300,
            'graph_type': graph_type,
            'degree': 2, 
            'noise_type': 'gaussian',
            # 'miss_type': 'mcar',
            # 'miss_percent': 0.1,
            "sem_type": sem_type,
            "ev": False
        }

    if config_id in (11,12,13):
        config['miss_type'] = 'mcar'
    elif config_id in (14,15,16):
        config['miss_type'] = 'mar'
    elif config_id in (17,18,19):
        config['miss_type'] = 'mnar'
    
    if config_id in (11,14,17):
        config['miss_percent'] = 0.1
    elif config_id in (12,15,18):
        config['miss_percent'] = 0.3
    if config_id in (13,16,19):
        config['miss_percent'] = 0.5

    if config_id > 20: 
        config['miss_type'] = 'mcar'
        config['miss_percent'] = 0.1
    
    if config_id == 27: 
        config['num_vars'] = 20
    elif config_id == 28: 
        config['num_vars'] = 30
    elif config_id == 29: 
        config['num_vars'] = 40
    elif config_id == 30: 
        config['num_vars'] = 50
    elif config_id == 31: 
        config['num_vars'] = 100
    elif config_id == 32: 
        config['num_vars'] = 200

    sem_type = sem_type.upper()
    
    config['code'] = f'{sem_type}-{graph_type}{config_id}'
    
    return config

def get_data(config_id, graph_type, sem_type, version):

    if config_id > 10 and graph_type == 'REAL':
        config = get_real_config(config_id, graph_type, sem_type)
    elif graph_type != 'REAL':
        config = get_config(config_id, graph_type, sem_type)
    else:
        raise ValueError('Wrong config!')
    
    if graph_type == 'REAL': 
        dataset = RealDataset(n = config['num_obs'], d = config['num_vars'], 
                                config_code = config['code'],
                                miss_type = config['miss_type'], 
                                miss_percent = config['miss_percent'], sem_type = sem_type)
    else:

        dataset = SyntheticDataset(n = config['num_obs'], d = config['num_vars'], 
                            config_code = config['code'],
                            graph_type = config['graph_type'], 
                            degree = config['degree'], 
                            noise_type = config['noise_type'],
                            miss_type = config['miss_type'], 
                            miss_percent = config['miss_percent'], 
                            sem_type = config['sem_type'],
                            equal_variances = config['ev'],
                            version=version
                            )

    return dataset, config

if __name__ == '__main__':

    
    for config_id in range(21,25):
        for graph_type in ('REAL', ):
            for sem_type in ('dream1', 'dream2', 'dream3', 'dream5'):
                get_data(config_id, graph_type, sem_type, version='real')

    