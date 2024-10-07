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
                metric_basins_NF[ii].append(DEF.compute_metric_NF_W(basin=basin,metric=metric))
                metric_basins_NF_reg[ii].append(DEF.compute_metric_NF_reg(basin=basin,metric=metric))
            except:
                print('metric:%s-station:%s'%(metric_list[ii],basin))
#%% plotting options
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
from matplotlib import rcParams
rcParams['font.family'] = 'Calibri'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
#%%
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e']
for ii in range(0, len(metric_list)):
    data_temp=[np.asarray(metric_basins_NF[ii]).reshape((-1,1)),np.asarray(metric_basins_NF_reg[ii]).reshape((-1,1))]
    # Plot violin plots for data1 and data2
    parts = axes[ii].violinplot(data_temp, showmeans=False, showmedians=True)
    # Set unique colors for each set

    for i in range(len(metric_basins_NF[ii])):
        for partname in ('bodies', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
            if partname == 'bodies':
                parts[partname][i].set_facecolor(colors[i])
                parts[partname][i].set_facecolor(colors[i])
            else:
                parts[partname].set_color(colors[i])
                parts[partname].set_color(colors[i])
#%%

