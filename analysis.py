import pandas as pd
import os
dfs = []
for file in os.listdir('output'):
    if 'grid_search' not in file:
        continue
    
    dfs.append(pd.read_csv('output/' + file))
    
df = pd.concat(dfs)
df = df.reset_index(drop=True)

objective_means = df.groupby(by='name').objective.mean()
print(objective_means)
df['gain'] = df.apply(lambda row: row['objective'] - objective_means.loc[row['name']], axis=1)
print(df['gain'])

best = df.groupby(by=['temperature', 'alpha', 'gamma']).gain.agg(['mean', 'std']).reset_index()
print(best.columns)
print(best[best['mean'] == best['mean'].min()])


import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(data=df, x='temperature', y='gain', hue='alpha')
plt.show()
sns.boxplot(data=df, x='gamma', y='gain')
plt.show()
sns.scatterplot(
    data = best,
    x = 'temperature',
    y = 'mean', 
    hue = 'alpha', 
    style='gamma',
    size = 'std', 
    sizes=(20,200),
).set(
    xscale='log'
)
plt.show()