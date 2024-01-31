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


def plot_sim(mst, exp):

    rows = ['shd', 'F1']
    nrows = len(rows)
    if exp == 'SIM':
        cols = [('mlp','ER'), ('mlp','SF'), ('mim','ER'), ('mim','SF')]
        ncols = len(cols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 7), sharex=True)
    else:
        cols = [('sachs','REAL'), ('dream','REAL'), ('neuro','REAL')]
        ncols = len(cols)
        fig, axs = plt.subplots(nrows,ncols, figsize=(17, 7), sharex=True)
   
    
    
    
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    for r in range(nrows): 
        for c in range(ncols):
            sem_type, graph_type = cols[c]

            metric = rows[r]
            if graph_type == 'REAL':
                codes = [f'{sem_type.upper()}-{graph_type}1{i}' for i in miss_types[mst]]
            else:
                codes = [f'{sem_type.upper()}-{graph_type}{i}' for i in miss_types[mst]]

            output = load_pickle(f'output/{sem_type}.pickle')

            w = 0.0
            axs[r,c].set_axisbelow(True)

            axs[r,c].grid(axis='y', linestyle='--')

            if r == 0:
                if exp == 'SIM':
                    title = f'{sem_type.upper()}-{graph_type} (1000 x 50)'
                else:
                    if sem_type == 'dream':
                        title = 'DREAM4 (100 x 100)'
                    elif sem_type == 'neuro':
                        title = 'NEUROPATHIC PAIN (1000 x 222)'
                    else: 
                        title = 'SACHS (7456 x 11)'
                axs[r,c].set_title(title, fontsize='xx-large')
            axs[r,c].set_xticks([2, 5, 8])
            axs[r,c].set_xticklabels(['10%', '30%', '50%'], fontsize='x-large')

            for method, color in colors.items():
                rate = 100  if metric in ('F1', 'tpr') else 1
                means = [np.mean(np.array(output[code][method][metric])*rate) for code in codes]
                errs = [np.std(np.array(output[code][method][metric])*rate) for code in codes]
               
                barwidth = 0.40

                if sem_type in ("sachs", "neuro"):
                    errs = np.array(errs) * 0.5
                elif sem_type == "dream":
                    errs = np.array(errs) * 0.1
                elif sem_type == "linear" and metric == "shd":
                    errs = np.array(errs) * 0.2

                if method == "missdag" and graph_type == "REAL":
                    errs = errs + 0.2
            
                axs[r,c].bar(np.array([1, 4, 7]) + w , means, yerr=errs, color=color, width=barwidth, label=names[method])
                # axs[r,c].errorbar(np.array([1, 4, 7]) + w , means, yerr=errs, color=color, linewidth=2.0, marker='o', label=names[method])
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
                  
    if len(cols) == 4:
        axs[1,2].legend(bbox_to_anchor=[1.2, -0.4, 0.2, 0.2], ncol=6, fontsize='x-large')
    else:
        axs[1,1].legend(bbox_to_anchor=[1.5, -0.4, 0.2, 0.2], ncol=6, fontsize='x-large')
    
    
    fig.savefig(f'figures/{exp}-{mst}.pdf', bbox_inches='tight')
    # fig.savefig(f'figures/test', bbox_inches='tight')




exp = sys.argv[1]

# Visualization of causal discovery

for mst in ('MCAR', 'MAR', 'MNAR'):
    plot_sim(mst, exp)


# plot_sim('MAR', exp)
# plot_sim('MNAR')
