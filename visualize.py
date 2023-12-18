import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re

'''
Sem type: Linear, MLP, MIM, GP-ADD, REAL
Graph type: ER, SF, Real
Miss type: MCAR, MAR, MNAR 
Miss percent: 0.1, 0.3, 0.5
Metric: TPR, F1, SHD, MAE, RMSE
'''

def code_to_config(code):
    sem_type, code = code.split('-')
    config_id = re.findall(r"\d+", code)
    config_id = int(config_id[0])
    graph_type = code.replace(str(config_id), '')
    
    if config_id in (1,2,3):
        miss_type = 'MCAR'
    elif config_id in (4,5,6):
        miss_type = 'MAR'
    elif config_id in (7,8,9):
        miss_type = 'MNAR'
    
    if config_id in (1,4,7):
        miss_percent = 0.1
    elif config_id in (2,5,8):
        miss_percent = 0.3
    if config_id in (3,6,9):
        miss_percent = 0.5
    
    return sem_type, graph_type, miss_type, miss_percent

        

df = pd.read_csv('output/final.csv')
df['SEM'] = df['Code'].map(lambda x: code_to_config(x)[0])
df['Graph'] = df['Code'].map(lambda x: code_to_config(x)[1])
df['Mechanism'] = df['Code'].map(lambda x: code_to_config(x)[2])
df['Percent'] = df['Code'].map(lambda x: code_to_config(x)[3])


metric = 'tpr'
graph = 'ER'
miss_type = 'MCAR'
sub_df = df.loc[(df['Metric']==metric) & (df['Graph']==graph) & (df['Mechanism']==miss_type), ]
colors = {'OTM': "red", 
          'MissDAG': "blue", 
          "Mean Imputer": "green",
          "SK Imputer": "yellow",
          "RR Imputer": "orange", 
          "Iterative Imputer": "purple"}

fig = plt.figure()
for method, color in colors.items(): 
    plt.plot(sub_df['Percent'], sub_df[method], '-', marker='o', c=color, label=method)

plt.legend(loc="lower center", ncol=3)
plt.savefig('figures/test.png')