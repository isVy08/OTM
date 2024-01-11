import pandas as pd 
from utils.io import load_txt
import numpy as np
import matplotlib.pyplot as plt
from utils.io import load_pickle



# Block configurations
miss_types = {'MCAR': [1,2,3], 'MAR': [4,5,6], 'MNAR': [7,8,9]}
miss_percents = {0.1: [1,4,7], 0.3: [2,5,8], 0.5: [3,6,9]}



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



def plot_intro():
    colors = { "mean": "green", "sk": "orange", "iterative": "purple"}
    names = {'mean': 'Mean Imputer', 'sk': 'OT Imputer', 'iterative': 'Iterative Imputer'}

    mlp_output = extract_baseline({}, 'baseline_abs', 'baseline_abs_imputation', 'mlp')
    linear_output = extract_baseline({}, 'v1/baseline_linear', 'v1/baseline_linear_imputation', 'linear')
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
    fig.savefig(f'figures/intro.pdf')

def plot_linear():
    output = load_pickle(f'output/linear.pickle')
    colors = {'otm': "red", 
          'missdag': "blue", 
          "mean": "green",
          "sk": "grey",
          "lin-rr": "orange", 
          "iterative": "purple"}

    names = {'otm': 'OTM', 'missdag': 'MissDAG', 'mean': 'Mean Imputer', 
         'sk': 'OT Imputer (SK)', 'lin-rr': 'OT Imputer (RR)', 'iterative': 'Iterative Imputer'}


    fig, axs = plt.subplots(2,2, figsize=(12, 7), sharex=True)
    fig.tight_layout(pad=4.0, w_pad=1.0, h_pad=0.8)

    barwidth = 0.35
    for r in range(2): 
        for c in range(2): 
            if c == 0:
                graph_type = 'ER'
            else:
                graph_type = 'SF'
            metric = 'shd' if r == 0 else 'F1'
            codes = [f'Linear-{graph_type}{i}' for i in miss_percents[0.1]]

            w = 0.0
            axs[r,c].set_axisbelow(True)
            axs[r,c].grid(axis='y', linestyle='--')

            if r == 0:
                axs[r,c].set_title(f'{graph_type} graph', fontsize='xx-large')
            axs[r,c].set_xticks([1.5, 4.5, 7.5])
            axs[r,c].set_xticklabels(['MCAR', 'MAR', 'MNAR'])

            for method, color in colors.items():
                rate = 100  if metric == 'F1' else 1
                means = [np.mean(np.array(output[code][method][metric])*rate) for code in codes]
                errs = [np.std(np.array(output[code][method][metric])*rate) * 0.2 for code in codes]
               
                axs[r,c].bar(np.array([1, 4, 7]) + w , means, yerr=errs, color=color, width=barwidth, label=names[method])
                w += barwidth

            if c == 0: 
                if metric in ('F1', 'tpr'):
                    metric_name = f'{metric} (%)'
                else:
                    metric_name = metric.upper()
                axs[r,c].set_ylabel(metric_name, fontsize='x-large')
                
    
    axs[1,1].legend(bbox_to_anchor=[0.5, -0.30, 0.2, 0.2], ncol=3, fontsize='x-large')
    fig.savefig(f'figures/linear.pdf', bbox_inches='tight')


def plot_runtime():
    data = load_txt(f'output/runtime.txt')
    xs = []
    otm = []
    missdag = []
    # dagma = []
    config = {'27':20, '28':30, '29':40, '30':50, '31':100, '32':200}
    for line in data:
        if 'MLP-ER' in line:
            code, time = line.split(': ')
            method, _, code = code.split('-')
            
            if method == 'OTM':
                otm.append(float(time) * 23000)
                xs.append(config[code[-2:]])
            elif method == 'MissDAG':
                missdag.append(float(time) * 10)
            # else:
            #     dagma.append(float(time) * 23000)
    plt.plot(xs, otm, color='red', label='OTM', marker='o', linewidth=2.0)
    plt.plot(xs, missdag, color='blue', label='MissDAG', marker='o', linewidth=2.0)
    # plt.plot(xs, dagma, color='green', label='DAGMA', marker='o', linewidth=2.0)
    plt.xticks(ticks=xs, labels=xs)
    plt.legend()
    plt.xlabel('Number of nodes')
    plt.ylabel('Training time (seconds)')
    plt.savefig('figures/runtime.pdf')

plot_linear()
# plot_intro()