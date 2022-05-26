import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import os
import matplotlib.ticker as mticker


def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"


base_results_df_path = os.path.join("..", "results", "auto_hp_tuning")+'/results.csv'

res_df = pd.read_csv(base_results_df_path, index_col=0)
res_df = res_df[['if_trained', 'win_ratio', 'episode_reward_mean', 'steps','config/gamma',
       'config/lr', 'config/n_hidden_layers', 'config/random_scaling']]


ax = plt.axes(projection='3d')

# Data for three-dimensional scattered points
zdata = res_df['win_ratio']
xdata = np.log10(res_df['config/gamma'])
ydata = np.log10(res_df['config/lr'])

ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis')

ax.set_xlabel("gamma")
ax.set_ylabel("learning rate")
ax.set_zlabel("win ratio")

ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle('Win ratio distribution in experiments:')
plt.show()


ax = plt.axes(projection='3d')

ax.scatter3D(xdata, ydata, zdata, c=res_df['config/n_hidden_layers'], cmap='viridis')

ax.set_xlabel("gamma")
ax.set_ylabel("learning rate")
ax.set_zlabel("win ratio")

ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle('Win ratio distribution in experiments:')
plt.show()