# -*- coding: utf-8 -*-
"""
First version: September 2024
@author: Mohammad Sina Jahangir (Ph.D.)
Email:mohammadsina.jahangir@gmail.com
#This code is for analysisng the hyper-paramters of the normalizing flow NF models
    ## Analysis is done for both daily and weekly modules

#Tested on Python 3.10
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
-pandas
-numpy
-matplotlib
"""
#%%import necessary libraries
import numpy as np
#importing plot functions
import matplotlib.pyplot as plt
from os import chdir
import pandas as pd
#%%
#change directory
    ##where the hyperparameters are saved for each catchment/basin
chdir(r'D:\Paper\Code\BayesianELM2')
#%%plotting options
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
#%%
#Daily assessment
#initialize  hyperparamters' lists
lstm_size=[]
dense_size=[]
dense_act=[]
#normalizing flow number of layers
nf_layers=[]
#normalizing flow layers' size
nf_size=[]

#populate the lists
for ii in range(0,421):
    #read csv file
    df=pd.read_csv(r'ResultsCamels_HybridNF\best_config_%d_lag_365_lead_7_daymet_v1.csv'%ii)
    
    #lstm hidden size
    lstm_size.append(int(df.iloc[0,1]))
    
    #dense size
    dense_size.append(int(df.iloc[1,1]))
    #dense activation
    dense_act.append(df.iloc[2,1])
    
    #NF properties
    nf_layers.append(int(df.iloc[3,1]))
    nf_size.append(int(df.iloc[4,1]))
#%%
#Weekly assessment
#initialize  hyperparamters' lists
lstm_size_w=[]
dense_size_w=[]
dense_act_w=[]
#normalizing flow number of layers
nf_layers_w=[]
#normalizing flow layers' size
nf_size_w=[]

#populate the lists
for ii in range(0,421):
    #read csv file
    df=pd.read_csv(r'ResultsCamels_HybridNF_Weekly\best_config_%d_lag_365_lead_7_daymet_v1.csv'%ii)
    
    #lstm hidden size
    lstm_size_w.append(int(df.iloc[0,1]))
    
    #dense size
    dense_size_w.append(int(df.iloc[1,1]))
    #dense activation
    dense_act_w.append(df.iloc[2,1])
    
    #NF properties
    nf_layers_w.append(int(df.iloc[3,1]))
    nf_size_w.append(int(df.iloc[4,1]))
#%%
#obtain the percentage of linear or elu
    #daily
lin_p=len(list(filter(lambda x:x=='linear',dense_act)))/len(dense_act)*100
elu_p=100-lin_p
    #weekly
lin_p_w=len(list(filter(lambda x:x=='linear',dense_act_w)))/len(dense_act_w)*100
elu_p_w=100-lin_p_w
#%%
# import library for CDF estimation
from statsmodels.distributions.empirical_distribution import ECDF
#%%
##empirical CDF derivation
ecdf_lstm = ECDF(lstm_size)
ecdf_lstm_w = ECDF(lstm_size_w)

ecdf_dense = ECDF(dense_size)
ecdf_dense_w = ECDF(dense_size_w)

ecdf_nfl = ECDF(nf_layers)
ecdf_nfl_w = ECDF(nf_layers_w)

ecdf_nfl_size = ECDF(nf_size)
ecdf_nfl_size_w = ECDF(nf_size_w)
#%%
#plot the ECDF
# Create the subplots grid

# colorcode for plotting
daily_c='#008080'
weekly_c='#B7410E'

fig, ax = plt.subplots(2, 2, figsize=(7,7),dpi=500,sharey='row')
fig.subplots_adjust(hspace=0.4, wspace=0.2)

# Add extra space below for the last centered subplot
fig.subplots_adjust(bottom=0.45)

    ##LSTM size (16-256)
ax[0,0].plot(ecdf_lstm.x, ecdf_lstm.y, label='Daily', color=daily_c, linestyle='-',linewidth=2)
ax[0,0].plot(ecdf_lstm_w.x, ecdf_lstm_w.y, label='Weekly', color=weekly_c, linestyle='--',linewidth=2)

ax[0,0].scatter(x=np.median(ecdf_lstm_w.x),y=0.5,c=weekly_c,s=42)
ax[0,0].scatter(x=np.median(ecdf_lstm.x),y=0.5,c=daily_c,s=42)

ax[0,0].scatter(x=np.median(ecdf_lstm_w.x),y=0,c=weekly_c,s=42)
ax[0,0].scatter(x=np.median(ecdf_lstm.x),y=0,c=daily_c,s=42)

# Add horizontal dashed lines at the median
max_median=max([np.median(ecdf_lstm.x),np.median(ecdf_lstm_w.x)])
min_median=min([np.median(ecdf_lstm.x),np.median(ecdf_lstm_w.x)])
ax[0,0].hlines(0.5, 0, max_median, colors=weekly_c, linestyles='--', linewidth=1.5)
ax[0,0].hlines(0.5, min_median, max_median, colors=daily_c, linestyles='--', linewidth=1.5)
ax[0,0].axvline(np.median(ecdf_lstm.x), 0, 0.5, color=daily_c, linestyle='--', linewidth=1.5)
ax[0,0].axvline(np.median(ecdf_lstm_w.x), 0, 0.5, color=weekly_c, linestyle='--', linewidth=1.5)
# Set axis limits to start from 0
ax[0,0].set_xlim(left=16)
ax[0,0].set_ylim(bottom=0)
for axis in ['top','bottom','left','right']:
    ax[0,0].spines[axis].set_linewidth(2)
ax[0,0].set_xlabel('LSTM size')
ax[0,0].set_ylabel('ECDF')
# Create a dashed gray box for the legend
legend = ax[0,0].legend(loc='best')
legend.get_frame().set_linewidth(2)
legend.get_frame().set_linestyle('--')
legend.get_frame().set_edgecolor('gray')


    ## Dense head size (16-64)
ax[0,1].plot(ecdf_dense.x, ecdf_dense.y, label='Daily', color=daily_c, linestyle='-',linewidth=2)
ax[0,1].plot(ecdf_dense_w.x, ecdf_dense_w.y, label='Weekly', color=weekly_c, linestyle='--',linewidth=2)

ax[0,1].scatter(x=np.median(ecdf_dense_w.x),y=0.5,c=weekly_c,s=42)
ax[0,1].scatter(x=np.median(ecdf_dense.x),y=0.5,c=daily_c,s=42)

ax[0,1].scatter(x=np.median(ecdf_dense_w.x),y=0,c=weekly_c,s=42)
ax[0,1].scatter(x=np.median(ecdf_dense.x),y=0,c=daily_c,s=42)


# Add horizontal dashed lines at the median
max_median=max([np.median(ecdf_dense.x),np.median(ecdf_dense_w.x)])
min_median=min([np.median(ecdf_dense.x),np.median(ecdf_dense_w.x)])
ax[0,1].hlines(0.5, 0, max_median, colors=weekly_c, linestyles='--', linewidth=1.5)
ax[0,1].hlines(0.5, min_median, max_median, colors=daily_c, linestyles='--', linewidth=1.5)
ax[0,1].axvline(np.median(ecdf_dense.x), 0, 0.5, color=daily_c, linestyle='--', linewidth=1.5)
ax[0,1].axvline(np.median(ecdf_dense_w.x), 0, 0.5, color=weekly_c, linestyle='--', linewidth=1.5)
# Set axis limits to start from 0
ax[0,1].set_xlim(left=8)
ax[0,1].set_ylim(bottom=0)
for axis in ['top','bottom','left','right']:
    ax[0,1].spines[axis].set_linewidth(2)
ax[0,1].set_xlabel('Dense size')

   ##Number of NF layers in MADE(1-3)
bar_daily=[np.count_nonzero(np.asarray(nf_layers)==1)/len(nf_layers)*100,\
  np.count_nonzero(np.asarray(nf_layers)==2)/len(nf_layers)*100,np.count_nonzero(np.asarray(nf_layers)==3)/len(nf_layers)*100]
bar_weekly=[np.count_nonzero(np.asarray(nf_layers_w)==1)/len(nf_layers_w)*100,\
  np.count_nonzero(np.asarray(nf_layers_w)==2)/len(nf_layers_w)*100,np.count_nonzero(np.asarray(nf_layers_w)==3)/len(nf_layers_w)*100]
bar_width = 0.25
x = np.arange(1,4)
ax[1,0].bar(x - bar_width/2,bar_daily,bar_width,color=daily_c, edgecolor=daily_c, linewidth=1.5)
bars2=ax[1,0].bar(x + bar_width/2,bar_weekly,bar_width,color=weekly_c, edgecolor=None, linewidth=1.5)

# Manually add dashed borders to the red bars using rectangles
for bar in bars2:
    # Get the rectangle's coordinates
    x_coord = bar.get_x()
    y_coord = bar.get_y()
    width = bar.get_width()
    height = bar.get_height()
    
    # Create a dashed rectangle around the bar
    rect = plt.Rectangle((x_coord, y_coord), width, height, edgecolor=weekly_c, facecolor='none', linestyle='--', linewidth=2)
    ax[1,0].add_patch(rect)

# Add horizontal dashed lines at the median
for axis in ['top','bottom','left','right']:
    ax[1,0].spines[axis].set_linewidth(2)
ax[1,0].set_xlabel('NF layers')
ax[1,0].set_ylabel('Percentage (%)')

    ## Dense activation type: Linear and ELU
bar_daily=[lin_p,elu_p]
bar_weekly=[elu_p_w,elu_p_w]
bar_width = 0.25
x = np.arange(0,2)
ax[1,1].bar(x - bar_width/2,bar_daily,bar_width,color=daily_c, edgecolor=daily_c, linewidth=1.5)
bars2=ax[1,1].bar(x + bar_width/2,bar_weekly,bar_width,color=weekly_c, edgecolor=None, linewidth=1.5)

# Manually add dashed borders to the red bars using rectangles
for bar in bars2:
    # Get the rectangle's coordinates
    x_coord = bar.get_x()
    y_coord = bar.get_y()
    width = bar.get_width()
    height = bar.get_height()
    
    # Create a dashed rectangle around the bar
    rect = plt.Rectangle((x_coord, y_coord), width, height, edgecolor=weekly_c, facecolor='none', linestyle='--', linewidth=2)
    ax[1,1].add_patch(rect)

# Add horizontal dashed lines at the median
for axis in ['top','bottom','left','right']:
    ax[1,1].spines[axis].set_linewidth(2)
ax[1,1].set_xlabel('Dense activation')
ax[1,1].set_xticks(x)
ax[1,1].set_xticklabels(['linear','elu'])



# Create the centered subplot at the bottom
    ##NF layer size
ax_last = fig.add_axes([0.3, 0.05, 0.4, 0.3])
ax_last.plot(ecdf_nfl_size.x, ecdf_nfl_size.y, label='Daily', color=daily_c, linestyle='-',linewidth=2)
ax_last.plot(ecdf_nfl_size_w.x, ecdf_nfl_size_w.y, label='Weekly', color=weekly_c, linestyle='--',linewidth=2)

ax_last.scatter(x=np.median(ecdf_nfl_size_w.x),y=0.5,c=weekly_c,s=42)
ax_last.scatter(x=np.median(ecdf_nfl_size.x),y=0.5,c=daily_c,s=42)

ax_last.scatter(x=np.median(ecdf_nfl_size_w.x),y=0,c=weekly_c,s=42)
ax_last.scatter(x=np.median(ecdf_nfl_size.x),y=0,c=daily_c,s=42)


# Add horizontal dashed lines at the median
max_median=max([np.median(ecdf_nfl_size.x),np.median(ecdf_nfl_size_w.x)])
min_median=min([np.median(ecdf_nfl_size.x),np.median(ecdf_nfl_size_w.x)])
ax_last.hlines(0.5, 0, max_median, colors=weekly_c, linestyles='--', linewidth=1.5)
ax_last.hlines(0.5, min_median, max_median, colors=daily_c, linestyles='--', linewidth=1.5)
ax_last.axvline(np.median(ecdf_nfl_size.x), 0, 0.5, color=daily_c, linestyle='--', linewidth=1.5)
ax_last.axvline(np.median(ecdf_nfl_size_w.x), 0, 0.5, color=weekly_c, linestyle='--', linewidth=1.5)
# Set axis limits to start from 0
ax_last.set_xlim(left=8)
ax_last.set_ylim(bottom=0)
for axis in ['top','bottom','left','right']:
    ax_last.spines[axis].set_linewidth(2)
ax_last.set_xlabel('NF size')
ax_last.set_ylabel('ECDF')
#save figure
    #change directory
#plt.savefig(r'D:\Paper\Code\HDL\Results\Hyperparamters_NF_D_W_v1.png',pad_inches=0.1)



    
    
    
    
    