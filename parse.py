import pandas as pd 
from utils.io import load_txt
'''
Columns: Code, OTM, Mean, SK, RR, Iterative, MissDag
Rows: Fscore, Gscore, SDH
'''
baseline = load_txt('output/baseline_linear.txt')
output = {}
for line in baseline:
    if 'ER' in line or 'SF' in line: 
        code = line.split('-')[:2]
        code = '-'.join(code)
        if code not in output:
            output[code] = {}
        method = line.split('-')[2:]
        method = '-'.join(method)
        if method not in output[code]:
            output[code][method] = {}
    elif 'F1' in line or 'gscore' in line or 'shd' in line:
        v = line.split(' : ')[-1]
        m = line.split(' : ')[0]
        output[code][method][m] = v
    
otm = load_txt('output/otm_linear.txt')
missdag = load_txt('output/missdag_linear.txt')

for line in otm:
    if 'ER' in line or 'SF' in line: 
        curr_code = line
    elif 'F1' in line or 'gscore' in line or 'shd' in line:
        value = line.split(' : ')[-1]
        m = line.split(' : ')[0]
        if 'otm' not in output[curr_code]:
            output[curr_code]['otm'] = {}
        output[curr_code]['otm'][m] = value

for line in missdag:
    if 'ER' in line or 'SF' in line: 
        curr_code = line
    elif 'F1' in line or 'gscore' in line or 'shd' in line:
        value = line.split(' : ')[-1]
        m = line.split(' : ')[0]
        if 'missdag' not in output[curr_code]:
            output[curr_code]['missdag'] = {}
        output[curr_code]['missdag'][m] = value

code = []
otm = []
missdag = []
mean = []
sk = []
linrr = []
iterative = []
indexs = []


for key, value in output.items():
    
    for m in ('F1', 'gscore', 'shd'):
        code.append(key)
        otm.append(value['otm'][m])
        missdag.append(value['missdag'][m]) 
        mean.append(value['mean'][m]) 
        sk.append(value['sk'][m]) 
        linrr.append(value['lin-rr'][m]) 
        iterative.append(value['iterative'][m]) 
        indexs.append(m)

print(otm)

# df = pd.DataFrame(data={
#     'Metric': indexs,
#     'Code' : code, 
#     'OTM': otm, 
#     'MissDAG': missdag, 
#     'Mean Imputer': mean, 
#     'SK Imputer': sk,
#     'RR Imputer': linrr, 
#     'Iterative Imputer': iterative
# })


# # print(df.head(100))
# df.to_csv('output/final.csv', index = False)