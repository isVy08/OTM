import npds.CauAcc as acc
import numpy as np
import os, random
import npds.sampling as spl
from utils.io import load_pickle, write_pickle

seed = random.choice(range(10))
class RealDataset:
    def __init__(self,n, d, config_code, miss_type, miss_percent):
        self.n = n 
        self.d = d 
        self.data_path = f'./dataset/{config_code}.pickle'
        
        # Load true DAG
        self.B_bin = acc.load_graph_true_graph()

        if os.path.isfile(self.data_path):
            print('Loading data ...')
            self.B_bin, self.X_true, self.X = load_pickle(self.data_path)
        else: 
            print('Generating and Saving data ...')

            # Simulate data
            model = spl.load_pgm()
            nms = spl.get_var_nms()
            df_sim = spl.random_sample(model, nms, n)
            df_sim = spl.nam2num(df_sim)

            

            # Add missing values
            '''
            default missingness mechanism is Missing Cmpletely A Rndom (MCAR)
            mode: can be 'mcar', 'mar', or 'mnar'
            mcar_p: probability of having a missing value that is MCAR
            mar_p: probability of having a missing given different parent values. 
            The first is the probability of having a missing value when its parent value is lower than threshold; 
            The second is the probability of having a missing value when its parent value is higher than threshold.
            mnar_p: the same with mar_p
            '''        
            if miss_type == 'mcar':
                df_miss = spl.add_missing_data(df_sim, mode='mcar', seed=seed, mcar_p=miss_percent)
            elif miss_type == 'mar':
                df_miss = spl.add_missing_data(df_sim, mode='mar', seed=seed, mar_p=[1-miss_percent, miss_percent])
            elif miss_type == 'mnar':
                df_miss = spl.add_missing_data(df_sim, mode='mnar', seed=seed, mar_p=[1-miss_percent, miss_percent])
            else:
                raise ValueError('Unknown missing type!')

            if d < df_sim.shape[1]:
                sub_nodes = self.select_sub_graph(d)
                df_sim = df_sim.iloc[:, sub_nodes]
                df_miss = df_miss.iloc[:, sub_nodes]
                B_bin = self.B_bin[sub_nodes, :]
                B_bin = B_bin[:, sub_nodes]
                self.B_bin = B_bin
                

            self.X_true = df_sim.to_numpy()
            self.X = df_miss.to_numpy()
            print(self.X_true.shape, self.X.shape)
            package = (self.B_bin, self.X_true, self.X)
            write_pickle(package, self.data_path)

            
    def select_sub_graph(self, num_nodes): 
        selected = set()
        edges = np.argwhere(self.B_bin).tolist()
        random.shuffle(edges)
        for i, j in edges:
            selected.add(i)
            selected.add(j)
            if len(selected) >= num_nodes:
                break 
        return list(selected)
