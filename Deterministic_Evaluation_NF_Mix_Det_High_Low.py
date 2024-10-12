# -*- coding: utf-8 -*-
"""
First version: Oct 2024
@author: Mohammad Sina Jahangir (Ph.D.)
Email:mohammadsina.jahangir@gmail.com
#This code is for evaluating two probabilistic methods, NF, MG, and DET
    ## Assessing low and high flow percent bias


#Tested on Python 3.11
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
chdir(r'D:\Paper\Code\HDL\Paper code')
import DEF
#%%
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
path_to_dir='D:\Paper\Code\BayesianELM2\ProcessedCamels_AllSame'
filenames = listdir(path_to_dir)
#%%
metric_list=['flv','fhv']
#Mixture independent
metric_basins_hybrid10qmMI=[]
#NF
metric_basins_NF=[]
#deterministic
metric_basins_det=[]

#removing basins that training was not stable
basin_rm=['01606500','02092500','02112360','03237500','06339100','06431500','06601000',\
'06847900','06889200','07375000','08070200','08082700','08178880','08195000','08200000','09447800','09480000',\
    '05508805','02479300']

for ii in range(0, len(metric_list)):
    metric=metric_list[ii]
    metric_basins_NF.append([])
    metric_basins_hybrid10qmMI.append([])
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
                metric_basins_hybrid10qmMI[ii].append(DEF.compute_metric_DELM_hybrid(basin=basin,metric=metric,mode='10QmMI'))
            except:
                print('metric:%s-station:%s'%(metric_list[ii],basin))
#%%
#Plot CDFs
# import library for CDF estimation
metric_list_=['FLV','FLH']
fig,ax=plt.subplots(nrows=1, ncols=2,figsize=(6, 4),dpi=600,sharey='all')
from statsmodels.distributions.empirical_distribution import ECDF


for ii in range(0,len(metric_list_)):
    
    len_=len(metric_basins_hybrid10qmMI[ii])
    print(len_)
    
    metric_array_hybrid10qmMI=np.asarray(metric_basins_hybrid10qmMI[ii]).reshape((len_,7))
    metric_array_hybrid10qmMI=np.nanmedian(metric_array_hybrid10qmMI,axis=1)
    
    metric_array_NF=np.asarray(metric_basins_NF[ii]).reshape((len_,7))
    metric_array_NF=np.nanmedian(metric_array_NF,axis=1)
    
    metric_array_det=np.asarray(metric_basins_det[ii]).reshape((len_,7))
    metric_array_det=np.nanmedian(metric_array_det,axis=1)
    
    
    
    metric_list=[metric_array_NF,metric_array_det,metric_array_hybrid10qmMI]
        
    ecdf_a1 = ECDF(metric_list[0])
    ecdf_a2 = ECDF(metric_list[1])
    ecdf_a3 = ECDF(metric_list[2])
    # Plot the ECDFs
    ax[ii].plot(ecdf_a1.x, ecdf_a1.y, label='NF', color='blue', linestyle='-',linewidth=1.5)
    ax[ii].plot(ecdf_a2.x, ecdf_a2.y, label='DT', color='red', linestyle='--',linewidth=1.5)
    ax[ii].plot(ecdf_a3.x, ecdf_a3.y, label=r'MG', color='purple', linestyle=':',linewidth=1.5)
    
    # Add the simple step function at zero
    ax[ii].plot([np.min(metric_list), 0], [0, 0], color='black', linestyle='-', linewidth=1.5)  # Zero before zero
    ax[ii].plot([0, np.max(metric_list)], [1, 1], color='black', linestyle='-', linewidth=1.5,label='Ideal line')  # One after zero
    # Add vertical line connecting at x=0
    ax[ii].plot([0, 0], [0, 1], color='black', linestyle='-', linewidth=1.5)  # Vertical line at x=0

    # Customize the plot
    ax[ii].set_xlabel(metric_list_[ii])
    if ii==0:
        ax[ii].set_ylabel('CDF')
    if ii==1:
        ax[ii].set_xlim(xmax=50)
        ax[ii].set_xlim(xmin=-100)
    legend=plt.legend()
    frame = legend.get_frame()
    frame.set_edgecolor('black')  # Set the edge color
    frame.set_linestyle('dashed')  # Set the edge style
    frame.set_facecolor('#F08080')  # Set the background color


    ax[ii].tick_params(direction="inout", right=True, length=8)
        
    [x.set_linewidth(2) for x in ax[ii].spines.values()]

plt.tight_layout()
plt.savefig(r'D:\Paper\Code\HDL\Results\FLV_FLH_Daily_v1.png')
    

