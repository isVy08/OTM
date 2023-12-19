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


miss_types = {'MCAR': [1,2,3, 11,12,13], 'MAR': [4,5,6, 14,15,16], 'MNAR': [7,8,9, 17,18,19]}
miss_percents = {0.1: [1,4,7, 11,14,17], 0.3: [2,5,8, 12,15,18], 0.5: [3,6,9, 13,16,19]}

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



if sem_type == 'gp-add': sem_type = 'gpadd'

# Visualization of causal discovery
if sem_type == 'real':
    nrows = 2 
    rows = ['F1', 'tpr']
    cols = ['MCAR', 'MAR', 'MNAR']
else:
    rows = ['shd', 'F1', 'tpr']
    cols = ['MCAR', 'MAR', 'MNAR']

nrows = len(rows)
ncols = 3
fig, axs = plt.subplots(nrows,ncols, figsize=(13,6), sharex=True)

for r in range(nrows): 
    for c in range(ncols): 
        metric, mst = rows[r], cols[c]
        if sem_type == 'real':
            codes = [f'{sem_type.upper()}-{graph_type}{i}' for i in miss_types[mst] if i > 10]
        else:
            codes = [f'{sem_type.upper()}-{graph_type}{i}' for i in miss_types[mst]]
        for method, color in colors.items():
            if method != 'missdag':
                means = [np.mean(output[code][method][metric]) for code in codes]
                errs = [np.std(output[code][method][metric]) for code in codes]
                axs[r,c].errorbar([0.1, 0.3, 0.5], means, yerr=errs, c=color, marker='^', label=names[method], alpha=0.6)
                axs[r,c].grid(axis='both', color='0.95', linestyle='--')
                if c > 0:
                    axs[r,c].get_yaxis().set_visible(False)
                else: 
                    axs[r,c].set_ylabel(metric.upper())

                if r == 0:
                    axs[r,c].set_title(mst)
                

i = nrows - 1
if nrows == 3:
    axs[i,i].legend(bbox_to_anchor=[0.5, -0.4, 0.2, 0.2], ncol=6)
else:
    axs[i,i].legend(bbox_to_anchor=[2.0, -0.23, 0.2, 0.2], ncol=6)
# plt.tight_layout()
plt.savefig(f'figures/{sem_type}-{graph_type}.png')