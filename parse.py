import pandas as pd 
from utils.io import load_txt
import numpy as np
import sys
from utils.io import write_pickle


def extract_baseline(output, sem_type, version, root='output'):

    graph = load_txt(f'{root}/{version}/baseline_{sem_type}.txt')
    
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
            if v == 'nan': v = 0
            m = line.split(' : ')[0]
            if m in output[code][method]:
                output[code][method][m].append(float(v))
            else:
                output[code][method][m] = [float(v)]

    imputation = load_txt(f'{root}/{version}/baseline_{sem_type}_imputation.txt')
   
    sem_type = 'Linear' if sem_type == 'linear' else sem_type.upper()
   
    for line in imputation:
        if ('ER' in line or 'SF' in line or 'REAL' in line) and sem_type in line: 
            if 'GP-ADD' in line: 
                line = line.replace('GP-ADD', 'GPADD')
            code = line.split('-')[:2]
            code = '-'.join(code)
            method = line.split('-')[2:]
            method = '-'.join(method)
            if method not in output[code]:
                output[code][method] = {}
        elif 'MAE' in line: 
            mae = line.split(', ')[0].split(': ')[-1]
            rmse = line.split(', ')[1].split(': ')[-1]
            output[code][method]['MAE'] = [np.round(float(mae), 4)]
            output[code][method]['RMSE'] = [np.round(float(rmse), 4)]
    return output

def extract_otm_missdag(output, method, sem_type, version, root='output'):  
    graph = load_txt(f'{root}/{version}/{method}_{sem_type}.txt')


    for line in graph:

        if 'ER' in line or 'SF' in line or 'REAL' in line: 
            if 'GP-ADD' in line: 
                line = line.replace('GP-ADD', 'GPADD')
            code = line
            if code not in output:
                output[code] = {}

        elif 'F1' in line or 'tpr' in line or 'shd' in line or 'gscore' in line or 'precision' in line:
            v = line.split(' : ')[-1]
            if v == 'nan': v = 0
            m = line.split(' : ')[0]
            if method not in output[code]:
                output[code][method] = {}
            if m in output[code][method]:
                output[code][method][m].append(float(v))
            else:
                output[code][method][m] = [float(v)]
    
    return output

def collect(method, sem_type, seeds=(1,2,3,4,5), root='output'): 

    for i in seeds:
        version = f'v{i}'
        if i == 1:
            if method == 'baseline':
                output = extract_baseline({}, sem_type, version, root=root)
            else:
                output = extract_otm_missdag({}, method, sem_type, version, root=root)
        else: 
            if method == 'baseline':
                temp = extract_baseline({}, sem_type, version, root=root)
            else:
                temp = extract_otm_missdag({}, method, sem_type, version, root=root)
            # levels: code > method > metric = value
            for code, l1_val in temp.items():
                for mth, l2_val in l1_val.items():  
                    for metric, value in l2_val.items():
                        output[code][mth][metric].extend(value)
    return output


if __name__ == "__main__":

# Combine output
    sem_type = sys.argv[1]
    if 'dream' in sem_type:
        version = 'real'
        output = extract_baseline({}, sem_type, version)
        output = extract_otm_missdag(output, 'otm', sem_type, version)
        output = extract_otm_missdag(output, 'missdag', sem_type, version)

    elif sem_type == 'linear':
        output = collect('otm', sem_type, (1,2,3))
        output_baseline = collect('baseline', sem_type, (1,2,3))
        output_missdag = collect('missdag', sem_type, (1,2,3))


        for code in output:
            for method in output_baseline[code]:
                output[code][method] = output_baseline[code][method]    
            for method in output_missdag[code]:
                output[code][method] = output_missdag[code][method]
    
    elif sem_type == 'ablation':
        output = collect('otm', 'mlp', seeds=(1,2,3,4,5), root='output/ablation')

        output_baseline = collect('baseline', 'mlp', seeds=(1,2,3,4,5), root='output/ablation')
        output_missdag = collect('missdag', 'mlp', seeds=(1,2,3), root='output/ablation')


        for code in output:
            for method in output_baseline[code]:
                output[code][method] = output_baseline[code][method]
            if code in output_missdag:
                for method in output_missdag[code]:
                    output[code][method] = output_missdag[code][method]
    else:
        output = collect('otm', sem_type)
        output_baseline = collect('baseline', sem_type)
        output_missdag = collect('missdag', sem_type)


        for code in output:
            for method in output_baseline[code]:
                output[code][method] = output_baseline[code][method]
            for method in output_missdag[code]:
                output[code][method] = output_missdag[code][method]

    write_pickle(output, f'output/{sem_type}.pickle')
