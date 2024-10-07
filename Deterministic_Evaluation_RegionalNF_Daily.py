# -*- coding: utf-8 -*-
"""
First version: Oct 2024
@author: Mohammad Sina Jahangir (Ph.D.)
Email:mohammadsina.jahangir@gmail.com
#This code is for evaluating regional NF model for seven-day ahead forecasting
    ## As no static features was used as input, it was not possible to train a
    "perfect" regional model
    ## A warm start was used and models were re-trained for each basin


#Tested on Python 3.9
Copyright (c) [2024] [Mohammad Sina Jahangir]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

#Dependencies:
-matplotlib
-pandas
-numpy
"""
#%%import necessary libraries
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from os import chdir
from os import listdir
import numpy as np
#change directory
#this is the folder where evaluation functions (DEF) are located
    ##change
chdir(r'D:\Paper\Code\BayesianELM2\Python code')
import DEF
#%%
#regional NF daily files
path_to_dir=r'D:\Paper\Code\BayesianELM2\ResultsCamels_HybridNF_Regional'
filenames = listdir(path_to_dir)
# Filter filenames to include only those with 'median' in their name
median_files = [filename for filename in filenames if 'median' in filename.lower()]
#%%
metric_list=['KGE','NSE']
#NF
metric_basins_NF=[]
#NF regional
metric_basins_NF_reg=[]

basin_rm=['01606500','02092500','02112360','03237500','06339100','06431500','06601000',\
'06847900','06889200','07375000','08070200','08082700','08178880','08195000','08200000','09447800','09480000',\
    '05508805','02479300']
for ii in range(0, len(metric_list)):
    metric=metric_list[ii]
    metric_basins_NF.append([])
    metric_basins_NF_reg.append([])
    for kk in range(0,len(median_files)):
        #not including the results of the basin_rm
        basin=median_files[kk][23:31]
        if basin in basin_rm:
            pass
        #try and catch for nan values!
        else:
            try:
                metric_basins_NF[ii].append(DEF.compute_metric_NF(basin=basin,metric=metric))
                metric_basins_NF_reg[ii].append(DEF.compute_metric_NF_reg(basin=basin,metric=metric))
            except:
                print('metric:%s-station:%s'%(metric_list[ii],basin))
#%% plotting options
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
from matplotlib import rcParams
rcParams['font.family'] = 'Calibri'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
#%%
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(7, 5),sharex=True,sharey=True,dpi=400)
# Define colors for the violins
colors = ['#1f77b4', '#d62728'] # Dark Blue and Dark Red
for ii in range(0, len(metric_list)):
    metric_array_NF=np.asarray(metric_basins_NF[ii]).reshape((len(metric_basins_NF[ii]),7))
    metric_array_NF_reg=np.asarray(metric_basins_NF_reg[ii]).reshape((len(metric_basins_NF_reg[ii]),7))
    list_array=[np.median(metric_array_NF,axis=1),np.median(metric_array_NF_reg,axis=1)]
    # Plot violin plots for data1 and data2
    parts = axes[ii].violinplot(list_array, showmeans=False, showmedians=True)
    # Assign different colors and increase linewidth for each violin
    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[idx % len(colors)])  # Alternate between Dark Blue and Dark Red
        pc.set_edgecolor('black')     # Set violin border color to black
        pc.set_linewidth(2)           # Increase the border thickness to 3 for emphasis
        pc.set_alpha(0.8)

    # Customize the median line to be bold and black
    for partname in ('cmedians','cbars', 'cmins', 'cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(2)  # Make the median lines bold
        
    # Display the median value on top of the cmaxes (upper whiskers)
    for idx, (median_value, cmax_value) in enumerate(zip(list_array, parts['cmaxes'].get_paths())):
        max_value = np.max(cmax_value.vertices[:, 1])  # Get the maximum y-value of the cmaxes (upper whiskers)
        median_value_median = np.median(median_value)  # Calculate the median value
        axes[ii].text(idx + 1, max_value + 0.02, f'{median_value_median:.2f}', 
                      horizontalalignment='center', fontsize=14, fontweight='bold')  # Add text above the cmaxes

    # Only show two labels for both axes
    axes[ii].set_xticks([1, 2])
    axes[ii].set_xticklabels(['NF', 'NF-reg'])  # Example labels, adjust as needed
    axes[0].set_ylabel('Seven-day median accuracy [-]')
    axes[ii].set_xlabel('Model')
    axes[ii].set_title(metric_list[ii])
# Save the plot
plt.tight_layout()
    #change directory
plt.savefig(r'D:\Paper\Code\HDL\Results\Compare_NF_Regional_Daily_v1.png')
#%%

