import pandas as pd 
from utils.io import load_txt
import numpy as np
import matplotlib.pyplot as plt
from utils.io import write_pickle


def extract_baseline(output, graph_path, imp_path, sem_type):

    graph = load_txt(f'output/{graph_path}.txt')
    
    for line in graph:
        if 'ER' in line or 'SF' in line or 'REAL' in line: 
            if 'GP-ADD' in line: 
                line = line.replace('GP-ADD', 'GPADD')
            code = line.split('-')[:2]
            code = '-'.join(code)
            if code not in output:
                output[code] = {}
            method = line.split('-')[2:]
            method = '-'.join(method)
            if method not in output[code]:
                output[code][method] = {}
        elif 'F1' in line or 'tpr' in line or 'shd' in line or 'gscore' in line or 'precision' in line:
            v = line.split(' : ')[-1]
            m = line.split(' : ')[0]
            output[code][method][m] = [float(v)]

    imputation = load_txt(f'output/{imp_path}.txt')

    sem_type = 'Linear' if sem_type == 'linear' else sem_type.upper()
      
    for line in imputation:
        if ('ER' in line or 'SF' in line or 'REAL' in line) and sem_type in line: 
            if 'GP-ADD' in line: 
                line = line.replace('GP-ADD', 'GPADD')
            code = line.split('-')[:2]
            code = '-'.join(code)
            method = line.split('-')[2:]
            method = '-'.join(method)
        elif 'MAE' in line: 
            mae = line.split(', ')[0].split(': ')[-1]
            rmse = line.split(', ')[1].split(': ')[-1]
            output[code][method]['MAE'] = [np.round(float(mae), 4)]
            output[code][method]['RMSE'] = [np.round(float(rmse), 4)]
    return output


# Block configurations
miss_types = {'MCAR': [1,2,3], 'MAR': [4,5,6], 'MNAR': [7,8,9]}
miss_percents = {0.1: [1,4,7], 0.3: [2,5,8], 0.5: [3,6,9]}

colors = { "mean": "green",
          "sk": "grey",
          "iterative": "purple"}

names = {'mean': 'Mean Imputer', 'sk': 'OT Imputer', 'iterative': 'Iterative Imputer'}

def plot(linear_output, mlp_output):
    fig, axs = plt.subplots(2,2, figsize=(12, 7), sharex=True)
    fig.tight_layout(pad=4.0, w_pad=1.0, h_pad=0.8)

    barwidth = 0.50
    for r in range(2): 
        for c in range(2): 
            if c == 0:
                output = linear_output
                sem_type = 'Linear'
            else:
                output = mlp_output
                sem_type = 'MLP'
            output = linear_output if c == 0 else mlp_output
            metric = 'RMSE' if r == 0 else 'F1'
            codes = [f'{sem_type}-ER{i}' for i in miss_types['MCAR']]

            w = 0.0
            axs[r,c].set_axisbelow(True)
            axs[r,c].grid(axis='y', linestyle='--')

            if r == 0:
                if sem_type == 'Linear':
                    sem_name = 'Linear SCM'
                else: 
                    sem_name = 'Non-linear SCM'
                axs[r,c].set_title(sem_name, fontsize='xx-large')
            axs[r,c].set_xticks([1.2, 4.2, 7.2])
            axs[r,c].set_xticklabels(['10%', '30%', '50%'])

            for method, color in colors.items():
                rate = 100  if metric == 'F1' else 1
                means = [np.mean(np.array(output[code][method][metric])*rate) for code in codes]
               
                
                if sem_type in ("sachs", "neuro"):
                    if method == 'missdag':
                        errs = np.array(errs) + 1.2
                    else: 
                        errs = np.array(errs) * 0.5
                if sem_type == "dream":
                    errs = np.array(errs) * 0.1
               
                axs[r,c].bar(np.array([1, 4, 7]) + w , means, color=color, width=barwidth, label=names[method])
                w += barwidth

            if c == 0: 
                if metric in ('F1', 'tpr'):
                    metric_name = f'{metric} (%)'
                else:
                    metric_name = metric
                axs[r,c].set_ylabel(metric_name, fontsize='x-large')
                
    

    axs[1,1].legend(bbox_to_anchor=[0.5, -0.30, 0.2, 0.2], ncol=6, fontsize='x-large')
    plt.savefig(f'figures/test.png')


mlp_output = extract_baseline({}, 'baseline_abs', 'baseline_abs_imputation', 'mlp')
linear_output = extract_baseline({}, 'v1/baseline_linear', 'v1/baseline_linear_imputation', 'linear')
plot(linear_output, mlp_output)
