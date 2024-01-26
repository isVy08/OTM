import matplotlib.pyplot as plt
import numpy as np
from utils.io import load_pickle, load_txt
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

# Block configurations
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

# del colors['missdag']

def plot_main(rows, cols, sem_type, graph_type, kind):
    nrows = len(rows)
    ncols = len(cols)
    fig, axs = plt.subplots(nrows,ncols, figsize=(15, 7), sharex=True)
    fig.tight_layout(pad=4.0, w_pad=1.0, h_pad=0.8)
    for r in range(nrows): 
        for c in range(ncols): 
            metric, mst = rows[r], cols[c]

            if graph_type == 'REAL':
                codes = [f'{sem_type.upper()}-{graph_type}1{i}' for i in miss_types[mst]]    
            else:
                if sem_type == 'linear':
                    codes = [f'Linear-{graph_type}{i}' for i in miss_types[mst]]
                else:
                    codes = [f'{sem_type.upper()}-{graph_type}{i}' for i in miss_types[mst]]

            w = 0.0
            axs[r,c].set_axisbelow(True)

            axs[r,c].grid(axis='y', linestyle='--')

            if r == 0:
                axs[r,c].set_title(mst, fontsize='xx-large')
            axs[r,c].set_xticks([2, 5, 8])
            axs[r,c].set_xticklabels(['10%', '30%', '50%'], fontsize='x-large')

            for method, color in colors.items():
                rate = 100  if metric in ('F1', 'tpr') else 1
                means = [np.mean(np.array(output[code][method][metric])*rate) for code in codes]
                errs = [np.std(np.array(output[code][method][metric])*rate) for code in codes]
               
                barwidth = 0.35

                if sem_type in ("sachs", "neuro"):
                    errs = np.array(errs) * 0.5
                elif sem_type == "dream":
                    errs = np.array(errs) * 0.1
                elif sem_type == "linear" and metric == "shd":
                    errs = np.array(errs) * 0.2

                if method == "missdag" and sem_type != "linear":
                    errs = None
            
                axs[r,c].bar(np.array([1, 4, 7]) + w , means, yerr=errs, color=color, width=barwidth, label=names[method])
                w += barwidth
            if c == 0: 
                if metric == 'F1':
                    metric_name = 'F1 (%)'
                elif sem_type == 'neuro' and metric == 'RMSE':
                    metric_name = 'ACCURACY (%)'
                else: 
                    metric_name = metric.upper()
                axs[r,c].set_ylabel(metric_name, fontsize='x-large')
            
            if r == 1: 
                axs[r,c].set_xlabel("Missing rate", fontsize='x-large')
                  

    i = nrows - 1
    if nrows == 3:
        axs[i,i].legend(bbox_to_anchor=[0.82, -0.4, 0.2, 0.2], ncol=6, fontsize='x-large')
    else:
        axs[1,1].legend(bbox_to_anchor=[1.90, -0.4, 0.2, 0.2], ncol=6, fontsize='x-large')
    
    fig.savefig(f'figures/{sem_type}-{graph_type}-{kind}.pdf', bbox_inches='tight')




sem_type = sys.argv[1]

graph_type = sys.argv[2]
output = load_pickle(f'output/{sem_type}.pickle')
if sem_type == 'gp-add': sem_type = 'gpadd'

# Visualization of causal discovery


rows = ['shd', 'F1']
cols = ['MCAR', 'MAR', 'MNAR']

plot_main(rows, cols, sem_type, graph_type, 'SL')

# if 'missdag' in colors: del colors['missdag']
# rows = ['MAE', 'RMSE']
# cols = ['MCAR', 'MAR', 'MNAR']
# plot_main(rows, cols, sem_type, graph_type, 'MI')