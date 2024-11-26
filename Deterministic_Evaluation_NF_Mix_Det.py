# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 18:18:52 2022
@author: sinaj
This code is for evaluationg the "deterministic" accuracy of DL models
for 421 camels dataset
This code is for comparison of daily NF with some other becnhmark DL models
such as multivariate normal
All vars
Deterministic (DT) added
Added the single difference of DT with NF
"""
#%%import necessary libraries
import numpy as np
#importing plot functions
import matplotlib.pyplot as plt
from os import chdir
#change directory
#this is the folder where files are
chdir(r'D:\Paper\Code\BayesianELM2\Python code')
import DEF
#%%
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
#%%
from os import listdir
#%%
path_to_dir='D:\Paper\Code\BayesianELM2\ProcessedCamels_AllSame'
filenames = listdir(path_to_dir)
#%%
metric_list=['KGE','NSE','nRMSE','nMAE']
#Multivariate
metric_basins_hybrid10qm=[]
#Mixture independent
metric_basins_hybrid10qmMI=[]
#Mixture multivariate
metric_basins_hybrid10qmMM=[]
#NF
metric_basins_NF=[]
#deterministic
metric_basins_det=[]


basin_rm=['01606500','02092500','02112360','03237500','06339100','06431500','06601000',\
'06847900','06889200','07375000','08070200','08082700','08178880','08195000','08200000','09447800','09480000',\
    '05508805','02479300']

for ii in range(0, len(metric_list)):
    metric=metric_list[ii]
    metric_basins_NF.append([])
    metric_basins_hybrid10qm.append([])
    metric_basins_hybrid10qmMI.append([])
    metric_basins_hybrid10qmMM.append([])
    metric_basins_det.append([])
    for kk in range(0,len(filenames)):
        #not including the results of the basin_rm
        basin=filenames[kk][7:15]
        if basin in basin_rm:
            pass
        #try and catch for nan values!
        else:
            try:
                metric_basins_NF[ii].append(DEF.compute_metric_NF(basin=basin,metric=metric))
                metric_basins_det[ii].append(DEF.compute_metric_Det(basin=basin,metric=metric))
                metric_basins_hybrid10qm[ii].append(DEF.compute_metric_DELM_hybrid(basin=basin,metric=metric,mode='10Qm'))
                metric_basins_hybrid10qmMI[ii].append(DEF.compute_metric_DELM_hybrid(basin=basin,metric=metric,mode='10QmMI'))
                metric_basins_hybrid10qmMM[ii].append(DEF.compute_metric_DELM_hybrid(basin=basin,metric=metric,mode='10QmMM'))
            except:
                print('metric:%s-station:%s'%(metric_list[ii],basin))
#%%
models_list=['NF','DT','MG']
#%%Converting to array and removing two of the basins
# Initiating the plot
fig, ax = plt.subplots(
    nrows=len(metric_list),
    ncols=1,
    figsize=(9, 12),
    sharey="row",
    sharex="all",
    dpi=600,
)

from matplotlib.lines import Line2D
# Box settings
boxprops = dict(linestyle="-", linewidth=2, color="k")
medianprops = dict(linestyle="-", linewidth=2, color="red")
colors = ["tan", "pink", "lightblue"]

# Positioning for grouped boxplots
x = np.arange(1, 8)  # 7 forecast steps
bar_width = 0.2  # Width of each group
positions = [x + i * bar_width for i in range(3)]


# Loop through metrics
for ii in range(len(metric_list)):
    len_ = len(metric_basins_hybrid10qm[ii])
    print(len_)
    
    # Convert metrics to arrays
    metric_array_hybrid10qm = np.asarray(metric_basins_hybrid10qm[ii]).reshape((len_, 7))
    metric_array_hybrid10qmMI = np.asarray(metric_basins_hybrid10qmMI[ii]).reshape((len_, 7))
    metric_array_hybrid10qmMM = np.asarray(metric_basins_hybrid10qmMM[ii]).reshape((len_, 7))
    metric_array_NF = np.asarray(metric_basins_NF[ii]).reshape((len_, 7))
    metric_array_det = np.asarray(metric_basins_det[ii]).reshape((len_, 7))
    
    # Combine all arrays for grouping
    list_array = [
        metric_array_NF,
        metric_array_hybrid10qm,
        metric_array_hybrid10qmMI,
    ]
    
    for i, category_data in enumerate(list_array):
    # Transpose the data for each category to create boxplots for each group
        bp = ax[ii].boxplot(
            category_data,
            positions=positions[i],
            widths=bar_width,
            patch_artist=True,
            boxprops=dict(facecolor=colors[i], color="black", linewidth=2),
            medianprops=dict(color="red", linewidth=1.5),
            showfliers=False,
        )
        
        # Find the median for the group
        med = np.nanmedian(category_data,axis=0)
            # Calculate the top whisker value
        top_whisker = np.percentile(category_data, 75,axis=0) + 1.5 * (np.percentile(category_data, 75,axis=0) - np.percentile(category_data, 25,axis=0))  # 1.5 IQR rule
        for qq in range(0,7):
            if ii in range(0,2):
                ax[ii].text(positions[i][qq],# Correct position for each group in each category
                    1, # Adjust the position above the boxplot
                        np.round(med[qq],2),
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        rotation=45,
                        color="black",
                )
            else:
                ax[ii].text(positions[i][qq],# Correct position for each group in each category
                    top_whisker[qq], # Adjust the position above the boxplot
                        np.round(med[qq],2),
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        rotation=45,
                        color="black",
                        )
                
    # Draw horizontal y-tick lines
    for ytick in ax[ii].get_yticks():
      ax[ii].axhline(y=ytick, color='gray', linestyle='--', linewidth=1, alpha=0.4)

    # Label settings
    ax[ii].set_ylabel(metric_list[ii])
    ax[ii].set_xticks(x + (len(list_array) - 1) * bar_width / 2)  # Center x-ticks
    ax[ii].set_xticklabels([f"LT= {i}" for i in x], rotation=45, ha="right")

    # Adjust spines and ticks
    [spine.set_linewidth(2) for spine in ax[ii].spines.values()]
    ax[ii].tick_params(direction="inout", right=True, length=8)
    
    # Add legend
    legend_labels =models_list
    handles = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(models_list))]
    if ii==0:
        ax[ii].legend(handles, legend_labels,frameon=True, fontsize=10,edgecolor='black',loc="lower right", ncol=3)
        
        
   

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(r"D:\Paper\Code\HDL\Results\Grouped_Boxplot.png", dpi=600)
plt.close()


