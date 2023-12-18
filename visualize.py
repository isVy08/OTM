import matplotlib.pyplot as plt
import numpy as np
from utils.io import load_pickle
import sys

'''
Sem type: Linear, MLP, MIM, GP-ADD, REAL
Graph type: ER, SF, Real
Miss type: MCAR, MAR, MNAR 
Miss percent: 0.1, 0.3, 0.5
Metric: TPR, F1, SHD, MAE, RMSE

Visualize a block of 9 figures of 3 miss types & 3 metrics: each plot at different miss percent
Repeat the block for different sem types and graph types
'''


miss_types = {'MCAR': [1,2,3], 'MAR': [4,5,6], 'MNAR': [7,8,9]}
miss_percents = {0.1: [1,4,7], 0.3: [2,5,8], 0.5: [3,6,9]}

colors = {'otm': "red", 
          'missdag': "blue", 
          "mean": "green",
          "sk": "grey",
          "lin-rr": "orange", 
          "iterative": "purple"}

names = {'otm': 'OTM', 'missdag': 'MissDAG', 'mean': 'Mean Imputer', 
         'sk': 'OT Imputer (SK)', 'lin-rr': 'OT Imputer (RR)', 'iterative': 'Iterative Imputer'}

# Block configurations
sem_type = sys.argv[1]
graph_type = sys.argv[2]

# sem_type = 'mlp'
# graph_type = 'ER'
output = load_pickle(f'output/{sem_type}.pickle')

rows = ['shd', 'F1', 'tpr']
cols = ['MCAR', 'MAR', 'MNAR']

if sem_type == 'gp-add': sem_type = 'gpadd'
# Visualization of causal discovery
nrows = 2 
ncols = 3
fig, axs = plt.subplots(nrows,ncols, figsize=(13,6), sharex=True)

for r in range(nrows): 
    for c in range(ncols): 
        metric, mst = rows[r], cols[c]
        codes = [f'{sem_type.upper()}-{graph_type}{i}' for i in miss_types[mst]]
        for method, color in colors.items():
            
            means = [np.mean(output[code][method][metric]) for code in codes]
            errs = [np.std(output[code][method][metric]) for code in codes]
            axs[r,c].errorbar([0.1, 0.3, 0.5], means, yerr=errs, c=color, marker='o', label=names[method], alpha=0.5)
            axs[r,c].grid(axis='both', color='0.95', linestyle='--')
            if c > 0:
                axs[r,c].get_yaxis().set_visible(False)
            else: 
                axs[r,c].set_ylabel(metric.upper())

            if r == 0:
                axs[r,c].set_title(mst)
            

i = nrows//2
axs[i,i].legend(bbox_to_anchor=[2.0, -0.32, 0.2, 0.2], ncol=6)
# plt.tight_layout()
plt.savefig(f'figures/{sem_type}-{graph_type}.png')