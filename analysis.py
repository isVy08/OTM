import pandas as pd 
from utils.io import load_txt
import numpy as np
import matplotlib.pyplot as plt
from utils.io import load_pickle



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
    from parse import extract_otm_missdag
    local_colors = { "mean": "green", "sk": "orange", "iterative": "purple", "complete": None}
    local_names = {'mean': 'Mean Imputer', 'sk': 'OT Imputer', 'iterative': 'Iterative Imputer'}
    
    
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

            for method, color in local_colors.items():
                if method == 'complete':
                    if r == 1 and c == 0:
                        axs[r,c].hlines(y = 95, xmin=0.8, xmax=8, linestyle='--', label="Complete data")
                    elif r == 1 and c == 1:
                        axs[r,c].hlines(y = 80, xmin=0.8, xmax=8, linestyle='--', label="Complete data")
                else:
                    rate = 100  if metric == 'F1' else 1
                    means = [np.mean(np.array(output[code][method][metric])*rate) for code in codes]
                    errs = [2.0 for _ in codes] if metric == 'F1' else [0.5 * rate for _ in codes]
                

                    axs[r,c].bar(np.array([1, 4, 7]) + w , means, yerr=errs, color=color, width=barwidth, label=local_names[method])
                    w += barwidth
                

            if c == 0: 
                if metric in ('F1', 'tpr'):
                    metric_name = f'{metric} (%)'
                else:
                    metric_name = metric
                axs[r,c].set_ylabel(metric_name, fontsize='x-large')
            
            

    axs[1,1].legend(bbox_to_anchor=[0.7, -0.30, 0.2, 0.2], ncol=6, fontsize='x-large')
    fig.savefig(f'figures/intro.pdf')

def plot_linear():
    output = load_pickle(f'output/linear.pickle')
   
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

def extract_runtime(config):
    data = load_txt(f'output/runtime.txt')
    output = {name: [] for name in names}
    output['DAGMA'] = {}
    # config = {27:20, 28:30, 29:40} #, 30:50, 31:100, 32:200}
    
    for line in data:

        line = line.replace('-MLP', '/MLP')
        line = line.replace('-ER', '/ER')
       
        code, time = line.split(': ')
        method, _, code = code.split('/')
        config_id = int(code[-2:])
        if config_id in config:
            if method == 'otm':
                output['otm'].append((float(time) * 23000) / 3600)
            elif method == 'missdag':
                output['missdag'].append((float(time) * 10) / 3600)
            elif method == 'DAGMA':
                output['DAGMA'][config_id] = (float(time) * 230000)

    for line in data: 
        line = line.replace('-MLP', '/MLP')
        line = line.replace('-ER', '/ER')
        code, time = line.split(': ')
        method, _, code = code.split('/')
        config_id = int(code[-2:])
        if config_id in config:
            if method not in ('otm', 'missdag', 'DAGMA'):
                time = (output['DAGMA'][config_id] + float(time)) / 3600
                output[method].append(time)
    
    return output

def plot_scalability():
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.tight_layout(pad=4.5, w_pad=1.0, h_pad=0.8)
    config = {27:20, 28:30, 29:40} #, 30:50, 31:100, 32:200}
    runtime = extract_runtime(config)
    ablation = load_pickle('output/ablation.pickle')

    xs = list(config.values())
    for method in colors: 
        axs[0].plot(xs, runtime[method], color=colors[method], marker='o', linewidth=2.0, label=names[method])
        # plt.fill_between(xs, np.array(output[method])+0.5, np.array(output[method])-0.5, facecolor=colors[method], alpha=0.5)

        axs[0].set_xticks(xs) 
        axs[0].set_xticklabels(xs)
        
        axs[0].set_title('Training time (hours)')
    
        axs[0].set_xlabel('Number of nodes')

    for c in (1,2):
        barwidth = 0.35
        codes = [f'MLP-ER{i}' for i in config]
        axs[c].set_xlabel('Number of nodes')

        w = 0.0
        axs[c].set_axisbelow(True)
        axs[c].grid(axis='y', linestyle='--')
        axs[c].set_xticks([1.5, 4.5, 7.5])
        axs[c].set_xticklabels(xs)

        if c == 1:
            metric = 'shd'
            metric_name ='SHD'
        else:
            metric = 'F1'
            metric_name ='F1 (%)'
        metric = 'shd' if c == 1 else 'F1'
        for method, color in colors.items():
            rate = 100  if metric == 'F1' else 1
            means = [np.mean(np.array(ablation[code][method][metric])*rate) for code in codes]
            errs = [np.std(np.array(ablation[code][method][metric])*rate) * 0.8 for code in codes]
            
            axs[c].bar(np.array([1, 4, 7]) + w , means, yerr=errs, color=color, width=barwidth, label=names[method])
            w += barwidth

        axs[c].set_title(metric_name, fontsize='x-large')
            
    
    axs[1].legend(bbox_to_anchor=[-1.15, -0.32, 0.2, 0.2], ncol=6, fontsize='x-large')
    fig.savefig(f'figures/scalability.pdf')
    # fig.savefig('figures/test.png')


def plot_quali():
    

    # mae, rmse, ot_xy, ot_xx, ot_yx, graph_metrics 
    output = load_pickle('output/quanti_behavior.pkl')
    metrics = ('RMSE', r'$W_2(\widehat{\mathbf{X}}, \mathbf{X})$', 'SHD', 'F1 (%)')
    pads = [0.6, 20, 7, 7]

    fig, axs = plt.subplots(2,2, figsize=(9, 6), sharex=True)
    fig.tight_layout(pad=4.0, w_pad=1.0, h_pad=2.0)
    i = 0
    xs = range(len(output))
    for r in range(2):
        for c in range(2):
            if i == 0:
                data = [item[1] for item in output]
            elif i == 1:
                data = [item[4] for item in output]
            elif i == 2: 
                data = [item[-1]['shd'] for item in output]
            else:
                data = [item[-1]['F1'] * 100 for item in output]
            
            
            axs[r,c].plot(xs, data, marker='o', linewidth=2.0, color="red")
            axs[r,c].set_title(metrics[i], fontsize='xx-large')
            axs[r,c].set_axisbelow(True)
            axs[r,c].grid(axis='y', linestyle='--')
            axs[r,c].fill_between(xs, np.array(data)+pads[i], np.array(data)-pads[i], facecolor='lightcoral', alpha=0.5)
            
            # ticks = list(range(0, 23000, 100))
            if r == 1:
                axs[r,c].set_xlabel(r'$t \times 10^2$' + ' (step)')
            
            # axs[r,c].set_xticklabels(ticks)

            i += 1
    # plt.savefig('figures/test.png')
    plt.savefig('figures/convergence.pdf')

def plot_ablation():
    # MCAR, 0.1
    output = load_pickle(f'output/ablation.pickle')
    mlp_output = load_pickle(f'output/mlp.pickle')
    output['MLP-ER1'] = mlp_output['MLP-ER1']
   
    fig, axs = plt.subplots(2,2, figsize=(15, 7))
    fig.tight_layout(pad=4.0, w_pad=1.0, h_pad=0.8)

    barwidth = 0.35
    
    for c in range(2): 
        if c == 0:
            codes = [f'MLP-ER{i}' for i in (1, 21, 22, 23)]
            lab = 'Noise type'
            ticks = ['Gaussian','Exponential', 'Laplace', 'Gumbel']
        elif c == 1:
            codes = [f'MLP-ER{i}' for i in (1, 24, 25, 26)]
            lab = 'Expected degree'
            ticks = ['2', '4', '6', '8']
        for r in range(2): 
                        

            w = 0.0
            axs[r,c].set_axisbelow(True)
            axs[r,c].grid(axis='y', linestyle='--')
            axs[r,c].set_xticks([1.5, 4.5, 7.5, 10.5])
            axs[r,c].set_xticklabels(ticks)
            if r == 0:
                axs[r,c].set_title(lab, fontsize='xx-large')
                        

            metric = 'shd' if r == 0 else 'F1'
            for method, color in colors.items():
                rate = 100  if metric == 'F1' else 1
                means = [np.mean(np.array(output[code][method][metric])*rate) for code in codes]
                errs = [np.std(np.array(output[code][method][metric])*rate) * 0.8 for code in codes]
               
                axs[r,c].bar(np.array([1, 4, 7, 10]) + w , means, yerr=errs, color=color, width=barwidth, label=names[method])
                w += barwidth

            if c == 0: 
                if metric in ('F1', 'tpr'):
                    metric_name = f'{metric} (%)'
                else:
                    metric_name = metric.upper()
                axs[r,c].set_ylabel(metric_name, fontsize='x-large')
                
    
    axs[1,1].legend(bbox_to_anchor=[0.80, -0.3, 0.2, 0.2], ncol=6, fontsize='x-large')
    fig.savefig(f'figures/ablation.pdf')



# plot_scalability()
plot_intro()
# plot_quali()
# plot_ablation()