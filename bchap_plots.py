# Load two .pkl files, convert into dataframes, and plot the data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import importlib


plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "CMM12"})

width_in_inches = 140 / 25.4
height_in_inches = 95 / 25.4

with open('G:/My Drive/1. TUWien/1. A RESEARCH/8. EXPERIMENTS/19_01_SWiT/results/Book_nlos_linear/Book_nlos_linear_all_dfs.pkl', 'rb') as f:
    all_dfs = pkl.load(f)

with open('G:/My Drive/1. TUWien/1. A RESEARCH/8. EXPERIMENTS/19_01_SWiT/results/Book_nlos_fine_tuner_Random/Book_nlos_fine_tuner_Random_all_dfs.pkl', 'rb') as f:
    all_dfs_RANDOM = pkl.load(f)

with open('G:/My Drive/1. TUWien/1. A RESEARCH/8. EXPERIMENTS/19_01_SWiT/results/Book_nlos_fine_tuner/Book_nlos_fine_tuner_all_dfs.pkl', 'rb') as f:
    all_dfs_tuner = pkl.load(f)

intensity_values = np.arange(0, 2, step=1) #from 6 to 4
# Define colors
# colors = [(0.1, 0.1, 0.1), # Black
#           (0.3, 0.6, 0.9), # Royal blue
#           (0.6, 0.3, 0.6), # Purple
#           #(0.9, 0.6, 0.3), # Orange
#           (0.5, 0.0, 0.0), # Maroon
#           #(0.9, 0.9, 0.3), # Dark Yellow
#           #(0.8, 0.2, 0.2), # Dark red
#             (0.3, 0.9, 0.9), # Cyan
#           #(0.1, 0.5, 0.5), # teal
#           (0.5, 0.5, 0.0)] # Olive
#           #(0.1, 0.8, 0.1)] # Dark green

colors = [(0.8, 0.2, 0.2),   # light red
          (0.2, 0.8, 0.2),   # light green
          (0.2, 0.2, 0.8),   # light blue
          (0.8, 0.8, 0.2),   # light yellow
          (0.8, 0.2, 0.8),   # light purple
          (0.2, 0.8, 0.8),   # light cyan
          (0.8, 0.5, 0.2),   # light orange
          (0.6, 0.6, 0.6)]   # light gray

markers = ['o', 'd','s']
# Create the main figure and axes
fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches))#plt.subplots(figsize=(5.5, 4.0))


color_index = 0
#for df_list, label in zip([all_dfs, all_dfs_RANDOM], ['SWiT+Fine-tuning', 'Fully-supervised']):
for i, (df_list, label) in enumerate(zip([all_dfs, all_dfs_RANDOM,all_dfs_tuner], ['SWiT+Linear', 'WiT-Fully-supervised','SWiT+Fine-tuning'])):    
    for df in df_list:
        # Get a unique epoch value from the 'epochs' column
        epochs = df['epochs'].unique()[0]
        # Plot the results
        ax.errorbar(intensity_values, df['avg_rmse'], yerr=1.96*df['std_rmse'], barsabove=False, 
                 marker=markers[i], mfc=colors[color_index], mec=colors[7], ms=8, mew=0.5, ecolor=colors[color_index], elinewidth=1.0, ls = "dashed", color=colors[color_index], capsize=5, label=f'{label}')
        color_index += 1

ax.set_xticks(np.arange(2))
ax.set_xticklabels(('0.1', '0.5'))
ax.set_ylabel("RMSE [mm]", fontsize = 13)
ax.set_xlabel("$\lambda_{\mathrm{data}}$", fontsize = 13)
# ylim between 0 and 0.5
ax.set_ylim([300, 950])
#ax.set_yscale('log')
plt.grid(True, axis = "y",which="both", ls="--", alpha=0.3)
from matplotlib import container
ax = plt.gca()
# Handle/Remove error bars from legend:
handles, labels = ax.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
ax.legend(handles, labels, loc=1, prop={'size': 12}, numpoints=1, ncol=1, fancybox=False)
#ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=2, prop={'size': 8})
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

# Save pdf and png
plt.savefig('G:/My Drive/1. TUWien/1. A RESEARCH/8. EXPERIMENTS/19_01_SWiT/results/Book_nlos_linear/Book_nlos_linear_tuner_random_lambda_size.pdf', bbox_inches='tight')
plt.savefig('G:/My Drive/1. TUWien/1. A RESEARCH/8. EXPERIMENTS/19_01_SWiT/results/Book_nlos_linear/Book_nlos_linear_tuner_random_lambda_size.png', bbox_inches='tight', dpi=300)

# # Finally, display the plot
plt.show()
