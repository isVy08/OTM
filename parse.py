import pandas as pd 
from utils.io import load_txt
import numpy as np
import sys
from utils.io import write_pickle
'''
Columns: Code, OTM, Mean, SK, RR, Iterative, MissDag
Rows: Fscore, Gscore, SDH
'''

def extract_baseline(output, sem_type, version):

    graph = load_txt(f'output/{version}/baseline_{sem_type}.txt')
    
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
        elif 'F1' in line or 'tpr' in line or 'shd' in line:
            v = line.split(' : ')[-1]
            m = line.split(' : ')[0]
            output[code][method][m] = [float(v)]

    imputation = load_txt(f'output/{version}/baseline_{sem_type}_imputation.txt')
   
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

def extract_otm_missdag(output, method, sem_type, version):  
    graph = load_txt(f'output/{version}/{method}_{sem_type}.txt')


    for line in graph:

        if 'ER' in line or 'SF' in line or 'REAL' in line: 
            if 'GP-ADD' in line: 
                line = line.replace('GP-ADD', 'GPADD')
            code = line
            if code not in output:
                output[code] = {}

        elif 'F1' in line or 'tpr' in line or 'shd' in line:
            value = line.split(' : ')[-1]
            m = line.split(' : ')[0]
            if method not in output[code]:
                output[code][method] = {}
            output[code][method][m] = [float(value)]
    
    if method == 'otm':
        imputation = load_txt(f'output/{version}/otm_{sem_type}_imputation.txt')
        sem_type = 'Linear' if sem_type == 'linear' else sem_type.upper()
        for line in imputation:
            if ('ER' in line or 'SF' in line or 'REAL' in line) and sem_type in line: 
                if 'GP-ADD' in line: 
                    line = line.replace('GP-ADD', 'GPADD')
                code = line
            elif 'MAE' in line: 
                mae = line.split(', ')[0].split(': ')[-1]
                rmse = line.split(', ')[1].split(': ')[-1]
                output[code][method]['MAE'] = [np.round(float(mae), 4)]
                output[code][method]['RMSE'] = [np.round(float(rmse), 4)]
    else:
        for line in graph:
            if 'ER' in line or 'SF' in line or 'REAL' in line:
                if 'GP-ADD' in line: 
                    line = line.replace('GP-ADD', 'GPADD')
                code = line
            else:
                output[code][method]['MAE'] = [0]
                output[code][method]['RMSE'] = [0]
    return output

def collect(method, sem_type): 

    for i in range(1,3):
        version = f'v{i}'
        if i == 1:
            if method == 'baseline':
                output = extract_baseline({}, sem_type, version)
            else:
                output = extract_otm_missdag({}, method, sem_type, version)
        else: 
            if method == 'baseline':
                temp = extract_baseline({}, sem_type, version)
            else:
                temp = extract_otm_missdag({}, method, sem_type, version)
            # levels: code > method > metric = value
            for code, l1_val in temp.items():
                for method, l2_val in l1_val.items():  
                    for metric, value in l2_val.items():
                        output[code][method][metric].extend(value)
    return output

def output_to_df(output, sem_type):
    code = []
    otm = []
    missdag = []
    mean = []
    sk = []
    linrr = []
    iterative = []
    complete = []
    metrics = []

    output = dict(sorted(output.items()))
    for key, value in output.items():
        
        for m in ('F1', 'tpr', 'shd', 'MAE', 'RMSE'):
            code.append(key)
            otm.append(value['otm'][m])

            if m in ('MAE', 'RMSE'):
                complete.append(0)
            else:
                complete.append(value['complete'][m])
            missdag.append(value['missdag'][m]) 
            mean.append(value['mean'][m]) 
            sk.append(value['sk'][m]) 
            linrr.append(value['lin-rr'][m]) 
            iterative.append(value['iterative'][m]) 
            metrics.append(m)

    df = pd.DataFrame(data={
        'Metric': metrics,
        'Code' : code, 
        'OTM': otm, 
        'MissDAG': missdag, 
        'Mean Imputer': mean, 
        'SK Imputer': sk,
        'RR Imputer': linrr, 
        'Iterative Imputer': iterative,
        'Complete': complete,
    })

    df.to_csv(f'output/{sem_type}.csv', index = False)


sem_type = sys.argv[1]
# sem_type = 'mlp'
output = collect('otm', sem_type)
output_baseline = collect('baseline', sem_type)
output_missdag = collect('missdag', sem_type)


# Combine output

for code in output:
    for method in output_baseline[code]:
        output[code][method] = output_baseline[code][method]
    for method in output_missdag[code]:
        output[code][method] = output_missdag[code][method]

write_pickle(output, f'output/{sem_type}.pickle')