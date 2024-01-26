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
    # from parse import extract_otm_missdag
    local_colors = { "mean": "green", "sk": "orange", "iterative": "purple", "complete": None}
    local_names = {'mean': 'Mean Imputer', 'sk': 'OT Imputer', 'iterative': 'Iterative Imputer', "complete": "Complete data"}

    output = load_pickle('output/mlp.pickle')
    from parse import extract_otm_missdag
    complete = extract_otm_missdag({}, 'complete', 'mlp', 'complete')
    fig, axs = plt.subplots(2,3, figsize=(15, 7), sharex=True)
    fig.tight_layout(pad=4.0, w_pad=1.0, h_pad=0.8)

    barwidth = 0.50
    for r in range(2): 
        for c in range(3): 
            if c == 0:
                mst = "MCAR"
            elif c == 1:
                mst = "MAR"
            else:
                mst = "MNAR"

            metric = 'RMSE' if r == 0 else 'F1'
            codes = [f'MLP-ER{i}' for i in miss_types[mst]]
            

            w = 0.0
            axs[r,c].set_axisbelow(True)
            axs[r,c].grid(axis='y', linestyle='--')

            if r == 0:
                axs[r,c].set_title(mst, fontsize='xx-large')
            axs[r,c].set_xticks([1.2, 4.2, 7.2])
            axs[r,c].set_xticklabels(['10%', '30%', '50%'], fontsize='xx-large')    

            for method, color in local_colors.items():
                rate = 100  if metric == 'F1' else 1
                if method == 'complete':
                    # means = [np.mean(np.array(complete[code][method][metric])*rate) for code in codes]
                    if r == 1:
                        axs[r,c].hlines(y = 95, xmin=0.8, xmax=8, linestyle='--', label="Complete data", linewidth=2.0)
                else:
                    means = [np.mean(np.array(output[code][method][metric])*rate) for code in codes]
                    errs = [np.std(np.array(output[code][method][metric])*rate) for code in codes]
                    axs[r,c].bar(np.array([1, 4, 7]) + w , means, yerr=errs, color=color, width=barwidth, label=local_names[method])
                    w += barwidth
                

            if c == 0: 
                if metric in ('F1', 'tpr'):
                    metric_name = f'{metric} (%)'
                elif metric == 'RMSE':
                    metric_name = r'$\Vert\widehat{\mathbf{X}} - \mathbf{X}\Vert_2$'
                else:
                    metric_name = metric
                axs[r,c].set_ylabel(metric_name, fontsize='xx-large')
            
            if r == 1: 
                axs[r,c].set_xlabel("Missing rate", fontsize='xx-large')
            
            

    axs[1,1].legend(bbox_to_anchor=[1.7, -0.42, 0.2, 0.2], ncol=4, fontsize='xx-large')
    # fig.savefig(f'figures/test.png', bbox_inches='tight')
    fig.savefig(f'figures/intro.pdf', bbox_inches='tight')

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
                axs[r,c].set_title(f'LGM-NV-{graph_type}', fontsize='xx-large')
            axs[r,c].set_xticks([1.5, 4.5, 7.5])
            axs[r,c].set_xticklabels(['MCAR', 'MAR', 'MNAR'], fontsize='xx-large')

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
                axs[r,c].set_ylabel(metric_name, fontsize='xx-large')
                
    
    axs[1,1].legend(bbox_to_anchor=[0.5, -0.32, 0.2, 0.2], ncol=3, fontsize='x-large')
    # axs[1,1].legend(bbox_to_anchor=[0.80, -0.32, 0.2, 0.2], ncol=6, fontsize='x-large')
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
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.tight_layout(pad=5.0, w_pad=1.0, h_pad=1.2)
    config = {27:20, 28:30, 29:40, 30:50, 31:100, 32:200}
    runtime = extract_runtime(config)
    ablation = load_pickle('output/ablation.pickle')

    xs = list(config.values())
    ax1 = plt.subplot(212)

    for method in colors: 
        ax1.plot(xs, runtime[method], color=colors[method], marker='o', linewidth=1.0, label=names[method])
        # plt.fill_between(xs, np.array(output[method])+0.5, np.array(output[method])-0.5, facecolor=colors[method], alpha=0.5)

        ax1.set_xticks(xs) 
        ax1.set_xticklabels(xs)
        
        ax1.set_ylabel('Training time (hours)', fontsize='large')
        ax1.set_xlabel('Number of nodes', fontsize='large')

    for metric in ('shd', 'F1'):
        # barwidth = 0.40
        codes = [f'MLP-ER{i}' for i in config]
        ax = plt.subplot(221) if metric == 'shd' else plt.subplot(222)
        # ax.set_xlabel('Number of nodes')

        w = 0.0
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--')
        ax.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5, 16.5])
        ax.set_xticklabels(xs, fontsize='large')

        if metric == 'F1':
            metric_name = 'F1 (%)'
        else: 
            metric_name = 'SHD'
        
        for method, color in colors.items():
            rate = 100  if metric == 'F1' else 1
            means = [np.mean(np.array(ablation[code][method][metric])*rate) for code in codes]
            errs = [np.std(np.array(ablation[code][method][metric])*rate) * 0.8 for code in codes]
            
            
            ax.errorbar(np.array([1, 4, 7, 10, 13, 16]) + w , means, yerr=errs, color=color, label=names[method], marker='o', linewidth=1.0)
            # w += barwidth

        ax.set_title(metric_name, fontsize='x-large')
            
    
    ax1.legend(bbox_to_anchor=[0.90, -0.45, 0.2, 0.2], ncol=3, fontsize='large')
    plt.savefig(f'figures/scalability.pdf', bbox_inches='tight')
    # plt.savefig('figures/test.png',bbox_inches='tight')


def plot_quali():
    

    # mae, rmse, ot_xy, ot_xx, ot_yx, graph_metrics 
    output = load_pickle('output/quanti_behavior.pkl')
    metrics = (r'$\Vert\widehat{\mathbf{X}} - \mathbf{X}\Vert_2$', r'$W_2(\widehat{\mathbf{X}}, \mathbf{X})$', 'SHD', 'F1 (%)')
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


            if r == 1:
                axs[r,c].set_xlabel(r'$t \times 10^2$' + ' (step)', fontsize = 'xx-large')
            
            axs[r,c].tick_params(axis='both', which='major', labelsize=12)
            # axs[r,c].set_xticklabels(ticks, fontsize='x-large')
            

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
            axs[r,c].set_xticklabels(ticks, fontsize = 'xx-large')
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
                axs[r,c].set_ylabel(metric_name, fontsize='xx-large')

            
    
    axs[1,1].legend(bbox_to_anchor=[0.80, -0.32, 0.2, 0.2], ncol=6, fontsize='x-large')
    fig.savefig(f'figures/ablation.pdf')



# plot_scalability()
# plot_intro()
# plot_quali()
# plot_ablation()
plot_linear()