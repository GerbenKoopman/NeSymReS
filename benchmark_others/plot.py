# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# df_clean = pd.read_csv('/home/scur1229/nesymres/benchmark_others/output/benchmark_accuracies.csv')
# df_noise = pd.read_csv('/home/scur1229/nesymres/benchmark_others/output/benchmark_accuracies_noise.csv')

# methods = df_clean['method'].unique()
# width = 0.35
# for method in methods:
#     df1 = df_clean[df_clean['method'] == method].reset_index(drop=True)
#     df2 = df_noise[df_noise['method'] == method].reset_index(drop=True)
#     if len(df1) == 0 or len(df2) == 0:
#         continue
#     x = np.arange(len(df1))
#     plt.figure(figsize=(20, 6))
#     plt.bar(x - width/2, df1['mse'], width, label='No Noise', color='skyblue')
#     plt.bar(x + width/2, df2['mse'], width, label='Noise', color='salmon')
#     plt.yscale('log')
#     plt.xlabel('Expression Index')
#     plt.ylabel('MSE (log scale)')
#     plt.title(f'{method}: MSE Comparison (Noise vs No Noise)')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'{method}_mse_comparison.png', dpi=300)
#     plt.close()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df_clean = pd.read_csv('/home/scur1229/nesymres/benchmark_others/output/benchmark_accuracies.csv')
# df_noise = pd.read_csv('/home/scur1229/nesymres/benchmark_others/output/benchmark_accuracies_noise.csv')

# df_clean['noise'] = 'No Noise'
# df_noise['noise'] = 'Noise'
# df_all = pd.concat([df_clean, df_noise])

# plt.figure(figsize=(10, 6))
# sns.violinplot(x='method', y='mse', hue='noise', data=df_all, split=True, scale='width', cut=0)
# plt.yscale('log')
# plt.ylabel('MSE (log scale)')
# plt.title('MSE Distribution Comparison (Violin Plot)')
# plt.tight_layout()
# plt.savefig('mse_violinplot_comparison.png', dpi=300)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# df_clean = pd.read_csv('/home/scur1229/nesymres/benchmark_others/output/benchmark_accuracies.csv')
# df_noise = pd.read_csv('/home/scur1229/nesymres/benchmark_others/output/benchmark_accuracies_noise.csv')

# methods = df_clean['method'].unique()
# data = []
# labels = []

# for method in methods:
#     mse_clean = df_clean[df_clean['method'] == method]['mse']
#     mse_noise = df_noise[df_noise['method'] == method]['mse']
#     data.extend([mse_clean, mse_noise])
#     labels.extend([f'{method}\nNo Noise', f'{method}\nNoise'])

# plt.figure(figsize=(10, 6))
# plt.boxplot(data, labels=labels, showfliers=False)
# plt.yscale('log')
# plt.ylabel('MSE (log scale)')
# plt.title('MSE Distribution Comparison (Box Plot)')
# plt.xticks(rotation=30)
# plt.tight_layout()
# plt.savefig('mse_boxplot_comparison.png', dpi=300)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# df_clean = pd.read_csv('/home/scur1229/nesymres/benchmark_others/output/benchmark_accuracies.csv')
# df_noise = pd.read_csv('/home/scur1229/nesymres/benchmark_others/output/benchmark_accuracies_noise.csv')

# methods = df_clean['method'].unique()
# for method in methods:
#     mse_clean = df_clean[df_clean['method'] == method]['mse'].values
#     mse_noise = df_noise[df_noise['method'] == method]['mse'].values
#     x = np.arange(len(mse_clean))
#     plt.figure(figsize=(16, 5))
#     plt.scatter(x, mse_clean, label='No Noise', alpha=0.7, s=20)
#     plt.scatter(x, mse_noise, label='Noise', alpha=0.7, s=20)
#     plt.yscale('log')
#     plt.xlabel('Expression Index')
#     plt.ylabel('MSE (log scale)')
#     plt.title(f'{method}: MSE Comparison (Noise vs No Noise)')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'{method}_mse_scatter_comparison.png', dpi=300)
#     plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_clean = pd.read_csv('/home/scur1229/nesymres/benchmark_others/output/benchmark_accuracies.csv')
df_noise = pd.read_csv('/home/scur1229/nesymres/benchmark_others/output/benchmark_accuracies_noise.csv')

methods = df_clean['method'].unique()
plt.figure(figsize=(16, 6))  

for method in methods:
    mse_clean = df_clean[df_clean['method'] == method]['mse'].values
    mse_noise = df_noise[df_noise['method'] == method]['mse'].values
    x = np.arange(len(mse_clean))
    diff = mse_noise - mse_clean
    plt.plot(x, diff, marker='o', linestyle='-', label=method, alpha=0.7)

plt.xlabel('Expression Index')
plt.ylabel('MSE Difference (Noise - No Noise)')
plt.title('MSE Difference Comparison (Noise vs No Noise)')
plt.legend()
plt.tight_layout()
plt.savefig('all_methods_mse_diff_line.png', dpi=300)
plt.show()