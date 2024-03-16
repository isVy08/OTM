import sys
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('output.csv')

df = pd.melt(df, id_vars=['Code', 'Metric', 'Noise', 'd'], value_vars=['SM','DAGMA','SMDG'])
df.rename(columns={"variable": "method"}, inplace=True)

df["sem"] =  df["Code"].map(lambda x: x.split('-')[0])

sem = sys.argv[1]

fig, axs = plt.subplots(2,3, figsize=(15, 7), sharex=True)
fig.tight_layout(pad=4.0, w_pad=1.0, h_pad=0.8)
noise_types = ['Gaussian', 'Gumbel', 'Laplace']

for r in range(2):
    m = 'shd' if r == 0 else 'F1'
    for c in range(3):
        if r == 0 and c == 0: 
            legend = 'auto'
        else: 
            legend = False
        data = df.loc[(df["Metric"]==m) & (df["Noise"]==noise_types[c]) & (df["sem"]==sem), ["d", "method", "value"]]
        sns.barplot(data=data, x="d", y="value", hue="method", errorbar=None, ax=axs[r,c], legend=legend)

        axs[r,c].set_ylabel(m.upper())

fig.savefig(f'{sem}.png',  bbox_inches='tight')
