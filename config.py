def get_config(config_id):
    configs = {
        1: {
            'num_obs': 1000,
            'num_vars': 100,
            'graph_type': 'SF',
            'degree': 4, 
            'noise_type': 'gaussian',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'linear',
            "ev": True
        }, 

        2: {
            'num_obs': 1000,
            'num_vars': 100,
            'graph_type': 'ER',
            'degree': 4, 
            'noise_type': 'gaussian',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'linear',
            "ev": False
        }, 
        
        3: {
            'num_obs': 200,
            'num_vars': 100,
            'graph_type': 'SF',
            'degree': 4, 
            'noise_type': 'gumbel',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'linear',
            "ev": False
        }, 

        4: {
            'num_obs': 200,
            'num_vars': 100,
            'graph_type': 'ER',
            'degree': 4, 
            'noise_type': 'laplace',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'mlp',
            "ev": True
        },

        5: {
            'num_obs': 200,
            'num_vars': 100,
            'graph_type': 'SF',
            'degree': 4, 
            'noise_type': 'uniform',
            'miss_type': 'mcar',
            'miss_percent': 0.1,
            "sem_type": 'gp',
            "ev": False
        }, 

    }

    return configs[config_id]
