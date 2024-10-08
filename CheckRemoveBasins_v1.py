# -*- coding: utf-8 -*-
"""
First version: Oct 2024
@author: Mohammad Sina Jahangir (Ph.D.)
Email:mohammadsina.jahangir@gmail.com
#This code is to check the basins seleted to be removed from the paper analaysis
    ##19 basins were removed as nan values were obtained for the forecasts/and or
    very large negative KGE/NSE values
    ##basins' IDs are presented below in the code; see basin_rm variable
    ## training for these basins was very unstable


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
import os
import numpy as np
import pandas as pd
#%%
#regional NF daily files
path_to_dir=r'D:\Paper\Code\BayesianELM2\ResultsCamels_HybridNF'
os.chdir(path_to_dir)
#%%
type_list=['Q','$\hat{Q}$']
#observations
obs=[]
#predictions (median)
pred=[]

LT='LT_%d'%(1)
#removed basin lists
basin_rm=['01606500','02092500','02112360','03237500','06339100','06431500','06601000',\
'06847900','06889200','07375000','08070200','08082700','08178880','08195000','08200000','09447800','09480000',\
    '05508805','02479300']
for ii in range(0, len(basin_rm)):
    df_med=pd.read_csv('forecast_median_camels_%s.csv'%(basin_rm[ii]))
    df_y=pd.read_csv('forecast_obs_camels_%s.csv'%(basin_rm[ii]))
    
    y=np.asarray(df_y.loc[:,'target']).reshape((-1,1))
    med=np.asarray(df_med.loc[:,LT]).reshape((-1,1))
    obs.append(y)
    pred.append(med)
    
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
#nash metric
def nse(observed, simulated):
    """
    Compute the Nash-Sutcliffe Efficiency (NSE) metric.
    
    Parameters:
    observed : array-like
        The observed or true values.
    simulated : array-like
        The simulated or predicted values.
        
    Returns:
    nse_value : float
        The Nash-Sutcliffe Efficiency (NSE) value.
    """
    # Convert inputs to numpy arrays for easier manipulation
    observed = np.array(observed)
    simulated = np.array(simulated)
    
    # Remove missing values
    mask = ~np.isnan(observed) & ~np.isnan(simulated)
    observed = observed[mask]
    simulated = simulated[mask]
    
    # Compute NSE
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    
    nse_value = 1 - (numerator / denominator)
    
    return nse_value
#%%
plt.style.use('classic')
# Create subplots
    ##Only 16 subplots are shown for illustrtative purposes
fig, axes = plt.subplots(4, 4, figsize=(16, 16),sharex=True,dpi=500)
# Define colors for the violins
colors = ['#1f77b4', '#d62728'] # Dark Blue and Dark Red
for ii in range(0,len(obs)-3):
    axes[ii%4,ii//4].plot(obs[ii],c=colors[0],lw=2,label=type_list[0])
    axes[ii%4,ii//4].plot(pred[ii],c=colors[1],lw=2,label=type_list[1])
    axes[ii%4,ii//4].set_title('Basin:%s-NSE:%1.2f'%(basin_rm[ii],nse(obs[ii],pred[ii])))
    if ii//4==0:
        axes[ii%4,ii//4].set_ylabel('discharge (mm)')
        legend=axes[ii%4,ii//4].legend()
        
    if ii%4==3:
        axes[ii%4,ii//4].set_xlabel('Sample')
        # setting ticks for x-axis 
        axes[ii%4,ii//4].set_xticks([0, 499, 999])
        axes[ii%4,ii//4].set_xticklabels([1, 500, 1000])
    #change axis linewidth
    for axis in ['top','bottom','left','right']:
        axes[ii%4,ii//4].spines[axis].set_linewidth(2)
    #tick paramters
    axes[ii%4,ii//4].tick_params(direction='inout', length=6, width=2, colors='k')
    axes[ii%4,ii//4].grid(axis='y', color='black', alpha=0.5)
plt.tight_layout()
plt.savefig(r'D:\Paper\Code\HDL\Results\CheckRemovedBasins_v1.png')
#%%

