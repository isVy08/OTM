import pandas as pd 
from utils.io import load_txt
import numpy as np
'''
Columns: Code, OTM, Mean, SK, RR, Iterative, MissDag
Rows: Fscore, Gscore, SDH
'''

def extract_baseline(output, sem_type='linear'):

    graph = load_txt(f'output/baseline_{sem_type}.txt')
    
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
            output[code][method][m] = v

    imputation = load_txt(f'output/baseline_{sem_type}_imputation.txt')
   
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
            output[code][method]['MAE'] = np.round(float(mae), 4)
            output[code][method]['RMSE'] = np.round(float(rmse), 4)
    return output

def extract_otm_missdag(output, method, sem_type):  
    graph = load_txt(f'output/{method}_{sem_type}.txt')


    for line in graph:
        if 'ER' in line or 'SF' in line or 'REAL' in line: 
            if 'GP-ADD' in line: 
                line = line.replace('GP-ADD', 'GPADD')
            code = line
        elif 'F1' in line or 'tpr' in line or 'shd' in line:
            value = line.split(' : ')[-1]
            m = line.split(' : ')[0]
            if method not in output[code]:
                output[code][method] = {}
            output[code][method][m] = value
    
    if method == 'otm':
        imputation = load_txt(f'output/otm_{sem_type}_imputation.txt')
        sem_type = 'Linear' if sem_type == 'linear' else sem_type.upper()
        for line in imputation:
            if ('ER' in line or 'SF' in line or 'REAL' in line) and sem_type in line: 
                if 'GP-ADD' in line: 
                    line = line.replace('GP-ADD', 'GPADD')
                code = line
            elif 'MAE' in line: 
                mae = line.split(', ')[0].split(': ')[-1]
                rmse = line.split(', ')[1].split(': ')[-1]
                output[code][method]['MAE'] = np.round(float(mae), 4)
                output[code][method]['RMSE'] = np.round(float(rmse), 4)
    else:
        for line in graph:
            if 'ER' in line or 'SF' in line or 'REAL' in line:
                if 'GP-ADD' in line: 
                    line = line.replace('GP-ADD', 'GPADD')
                code = line
            else:
                output[code][method]['MAE'] = 'NA'
                output[code][method]['RMSE'] = 'NA'
    return output

sem_type = 'real'
output = extract_baseline({}, sem_type)
output = extract_otm_missdag(output, 'otm', sem_type)
# output = extract_otm_missdag(output, 'missdag', sem_type)
# print(output)

code = []
otm = []
missdag = []
mean = []
sk = []
linrr = []
iterative = []
metrics = []

output = dict(sorted(output.items()))
for key, value in output.items():
    
    for m in ('F1', 'tpr', 'shd', 'MAE', 'RMSE'):
        code.append(key)
        otm.append(value['otm'][m])
        # missdag.append(value['missdag'][m]) 
        mean.append(value['mean'][m]) 
        sk.append(value['sk'][m]) 
        linrr.append(value['lin-rr'][m]) 
        iterative.append(value['iterative'][m]) 
        metrics.append(m)

df = pd.DataFrame(data={
    'Metric': metrics,
    'Code' : code, 
    'OTM': otm, 
    # 'MissDAG': missdag, 
    'Mean Imputer': mean, 
    'SK Imputer': sk,
    'RR Imputer': linrr, 
    'Iterative Imputer': iterative
})

df.to_csv('output/final.csv', index = False)

