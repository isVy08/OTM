import numpy as np
import os, random
import networkx as nx
import npds.sampling as spl
import npds.CauAcc as acc
from utils.data import produce_NA
from cdt.data import load_dataset
from utils.io import load_pickle, write_pickle


seed = random.choice(range(10))
class RealDataset:
    def __init__(self, n, d, config_code, miss_type, miss_percent, sem_type, opt="logistic"):
        self.n = n 
        self.d = d 
        self.data_path = f'./dataset/{config_code}.pickle'

        if os.path.isfile(self.data_path):
            print('Loading data ...')
            self.B_bin, self.X_true, self.X = load_pickle(self.data_path)
        else:
            print(f'Generating and Saving {sem_type} data ...')
            if sem_type == 'neuro':
                
                # Load true DAG
                self.B_bin = acc.load_graph_true_graph()
                
            
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

            

                self.X_true = df_sim.to_numpy()
                self.X = df_miss.to_numpy()
                
            
            elif sem_type == 'sachs':
                from sklearn.preprocessing import StandardScaler
                s_data, s_graph = load_dataset('sachs')
                scaler = StandardScaler()
                self.X_true = scaler.fit_transform(s_data)
                # self.X_true = s_data.to_numpy()
                self.X, self.mask = produce_NA(self.X_true, miss_percent, mecha=miss_type, opt=opt, p_obs=0.3, q=0.3)
                self.B_bin = nx.adjacency_matrix(s_graph).todense()
            
            elif 'dream' in sem_type: 
                if sem_type == 'dream4':
                    s_data, s_graph = load_dataset('dream4-2')
                elif sem_type == 'dream1':
                    s_data, s_graph = load_dataset('dream4-1')
                elif sem_type == 'dream2':
                    s_data, s_graph = load_dataset('dream4-4')
                elif sem_type == 'dream3':
                    s_data, s_graph = load_dataset('dream4-3')
                elif sem_type == 'dream5':
                    s_data, s_graph = load_dataset('dream4-5')
                self.X_true = s_data.to_numpy()
                self.X, self.mask = produce_NA(self.X_true, miss_percent, mecha=miss_type, opt=opt, p_obs=0.3, q=0.3)
                self.B_bin = nx.adjacency_matrix(s_graph).todense()
            
            else: 
                raise ValueError('Unknown dataset!')

            if d < self.X_true.shape[1]:
                sub_nodes = self.select_sub_graph(d)
                sub_nodes = sorted(sub_nodes)
                self.X_true = self.X_true[:, sub_nodes]
                self.X = self.X[:, sub_nodes]
                B_bin = self.B_bin[sub_nodes, :]
                B_bin = B_bin[:, sub_nodes]
                self.B_bin = B_bin
                    
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
