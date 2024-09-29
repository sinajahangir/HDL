# -*- coding: utf-8 -*-
"""
First version: Oct 2024
@author: Mohammad Sina Jahangir (Ph.D.)
Email:mohammadsina.jahangir@gmail.com
#This code is for developing a regional NF model for seven-day ahead forecasting

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
-numpy
-pandas
-tensorflow
-tensorflow_probability
"""

#importing necessary libs
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import time
import os
tfpl=tfp.layers
#%%
#function to convert time-series to 3D array of the LSTMs
def split_sequence_multi_train(sequence_x,sequence_y, n_steps_in, n_steps_out,mode='seq'):
    """
    written by:SJ
    sequence_x=features; 2D array
    sequence_y=target; 2D array
    n_steps_in=IL(lookbak period);int
    n_steps_out=forecast horizon;int
    mode:either single (many to one) or seq (many to many).
    This function creates an output in shape of (sample,IL,feature) for x and
    (sample,n_steps_out) for y
    """
    X, y = list(), list()
    k=0
    sequence_x=np.copy(np.asarray(sequence_x))
    sequence_y=np.copy(np.asarray(sequence_y))
    for _ in range(len(sequence_x)):
		# find the end of this pattern
        end_ix = k + n_steps_in
        out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
        if out_end_ix > len(sequence_x):
            break
		# gather input and output parts of the pattern
        seq_x = sequence_x[k:end_ix]
        #mode single is used for one output
        if n_steps_out==0:
            seq_y= sequence_y[end_ix-1:out_end_ix]
        elif mode=='single':
            seq_y= sequence_y[out_end_ix-1]
        else:
            seq_y= sequence_y[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y.flatten())
        k=k+1
    
    XX,YY= np.asarray(X), np.asarray(y)
    if (n_steps_out==0 or n_steps_out==1):
        YY=YY.reshape((len(XX),1))
    return XX,YY
#%%
#function to convert time-series to 3D array of the LSTMs
def split_sequence_multi_s(sequence_x,sequence_y, n_steps_in, n_steps_out,mode='seq'):
    """
    written by:SJ
    sequence_x=features; 2D array
    sequence_y=target; 2D array
    n_steps_in=IL(lookbak period);int
    n_steps_out=forecast horizon;int
    mode:either single (many to one) or seq (many to many).
    This function creates an output in shape of (sample,IL,feature) for x and
    (sample,n_steps_out) for y
    """
    X, y = list(), list()
    k=0
    sequence_x=np.copy(np.asarray(sequence_x))
    sequence_y=np.copy(np.asarray(sequence_y))
    for _ in range(len(sequence_x)):
		# find the end of this pattern
        end_ix = k + n_steps_in
        out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
        if out_end_ix > len(sequence_x):
            break
		# gather input and output parts of the pattern
        seq_x = sequence_x[end_ix:out_end_ix]
        #mode single is used for one output
        if n_steps_out==0:
            seq_y= sequence_y[end_ix-1:out_end_ix]
        elif mode=='single':
            seq_y= sequence_y[out_end_ix-1]
        else:
            seq_y= sequence_y[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y.flatten())
        k=k+1
    
    XX,YY= np.asarray(X), np.asarray(y)
    if (n_steps_out==0 or n_steps_out==1):
        YY=YY.reshape((len(XX),1))
    return XX,YY
#%%
#Z-normalization
def normalize(x,mean_,sd_):
  """
  Written by:SJ
  x:input;2D array
  This function z-normalizes the input
  returns:transformed data
  """
  x_norm=(x-mean_)/sd_
  return x_norm
#%%
#Reverse normalization
def i_normalize(x_tr,mean_,sd_):
  """
  Written by:SJ
  mean_,sd_,:mean_, and sd_ used for initial transformation; 2D array
  x_tr:transformed input
  This function inverses the transformation
  returns:inverse transfomed 
  """
  x_i=x_tr*sd_+mean_
  return x_i
#%%
def index_cal(y,train_r=0.8,val_r=0.1):
  """
  Written by:SJ
  y:target;2D vector
  train_r=training ratio;float
  val_r=training ratio;float
  This function claculates the last sample index for training, and validation
  returns: 
  """
  train_id=int(train_r*len(y))
  valid_id=int((val_r+train_r)*len(y))
  return train_id,valid_id
#%%
from os import listdir
#%%
path_to_dir='./Selected daymet'
filenames = listdir(path_to_dir)
#%%
#change directory
save_path_results='ResultsCamels_HybridNF_Regional'
if not os.path.isdir(save_path_results):
    os.mkdir(save_path_results)
#%%Read observation data function
def read_data(basin_id=0,LT=7,IL=365):
  """
  Written by: SJ
  basind_id=number of basin;int
  LT=forecast lead time;int
  IL=lookback period;int
  This function reads the data, z-normalizes it, divides it to training
  validation, and testing, and outputs it in a (sample,timestep,feature) format
  for x and (sample,LT) for y
  """
  #reading the data
  filename='./Selected daymet/%s'%(filenames[basin_id])
  #Q is the last column
  df_data=pd.read_csv(filename,header=0)
  #as we will be using in the simulation mode, lagged Q is not used as input
  #removing date (first column)
  x=np.asarray(df_data.iloc[:,1:])
  #removing SWE
  x_=np.copy(np.delete(x,2,1))
  train_id,val_id=index_cal(x_)
  x_train_=x_[:train_id]
  mean_=np.mean(x_train_,axis=0)
  sd_=np.std(x_train_,axis=0)
  
  #normalizing based on training data
  x_norm_=normalize(x_,mean_,sd_)
  y_norm_=np.copy(x_norm_[:,-1])
  
  #removing Q from input matrix
  #x_norm_=x_norm_[:,:-1]
  
  xx,yy=split_sequence_multi_train(x_norm_,y_norm_,IL,LT)

  #training set
  xx_train=xx[0:train_id-IL-LT+1,:]
  yy_train=yy[0:train_id-IL-LT+1,:]

  #validation set
  xx_val=xx[train_id-IL-LT+1:val_id-IL-LT+1,:]
  yy_val=yy[train_id-IL-LT+1:val_id-IL-LT+1,:]

  #test set
  xx_test=xx[val_id-IL-LT+1:,:]
  yy_test=yy[val_id-IL-LT+1:,:]

  return [xx_train,xx_val,xx_test],[yy_train,yy_val,yy_test]
#%%
#read forecast data
def read_data_s(basin_id=0,LT=7,IL=365):
  """
  Written by: SJ
  basind_id=number of basin;int
  LT=forecast lead time;int
  IL=lookback period;int
  This function reads the data, z-normalizes it, divides it to training
  validation, and testing, and outputs it in a (sample,timestep,feature) format
  for x and (sample,LT) for y
  """
  #reading the data
  filename='./ProcessedCamels_AllSame/%s'%(filenames[basin_id])
  #Q is the last column
  df_data=pd.read_csv(filename,header=0)
  #as we will be using in the simulation mode, lagged Q is not used as input
  #removing date (first column)
  x=np.asarray(df_data.iloc[:,1:])
  #removing zeros
      #SWE
  x_=np.copy(np.delete(x,13,1))
  train_id,val_id=index_cal(x_)
  x_train_=x_[:train_id]
  mean_=np.mean(x_train_,axis=0)
  sd_=np.std(x_train_,axis=0)
  
  #normalizing based on training data
  x_norm_=normalize(x_,mean_,sd_)
  y_norm_=np.copy(x_norm_[:,-1])
  
  #removing Q from input matrix
  x_norm_=x_norm_[:,:-1]
 
  xx,yy=split_sequence_multi_s(x_norm_,y_norm_,IL,LT)

  #training set
  xx_train=xx[0:train_id-IL-LT+1,:]
  yy_train=yy[0:train_id-IL-LT+1,:]

  #validation set
  xx_val=xx[train_id-IL-LT+1:val_id-IL-LT+1,:]
  yy_val=yy[train_id-IL-LT+1:val_id-IL-LT+1,:]

  #test set
  xx_test=xx[val_id-IL-LT+1:,:]
  yy_test=yy[val_id-IL-LT+1:,:]

  return [xx_train,xx_val,xx_test],[yy_train,yy_val,yy_test]
#%%
"""# Model Develpment"""
tfb = tfp.bijectors
#LSTM forecast module
def model_lstm_(n_features=37,lstm_out_dec=256,dense_out_dec=8,\
        num_components = 2,component_value=32,activation_dense_dec='elu',nout=7,nin=365,n_features_era=5):
  """
  Written by:SJ
  lstm_out:Output of lstm;int
  dense_out:Output of dense layer;int
  activation_dense_i: ith dense layer activation function
  The models is consisted of a LSTM+DL+distribution output
  Multivariate normal distribution is considered for the output
  returns:compiled tf model
  """
  latent_dim_lstm=lstm_out_dec
  latent_dim_dense=dense_out_dec
  
  
  input_x=tf.keras.Input(shape=(nin, n_features),name='Input_x')
  #
  input_y=tf.keras.Input(shape=(nout, n_features_era),name='Input_y')
  ##encoder-the LSTM
  encoder = tf.keras.layers.LSTM(latent_dim_lstm, return_state=True,name='enc_lstm')
  encoder_outputs, state_h, state_c = encoder(input_x)
  
  ##decoder
  decoder = tf.keras.layers.LSTM(latent_dim_lstm,name='dec_lstm')
  decoder_outputs = decoder(input_y)
  
  #concat
  concat_=tf.keras.layers.Concatenate(name='concat')([decoder_outputs,encoder_outputs])
  decoder_ = tf.keras.layers.Dense(latent_dim_dense,activation=activation_dense_dec,name='dec_dense')
  x_ = decoder_(concat_)
  
  ##probabilistic layer
  params_size = tfpl.IndependentNormal.params_size(nout)
  x_=tf.keras.layers.Dense(params_size,activation=activation_dense_dec,name='dec_prob')(x_)
  
  y_=tfpl.IndependentNormal(event_shape=nout,convert_to_tensor_fn=tfp.distributions.Distribution.sample,name='output')(x_)
  
  layer_nf=[int(component_value)] * num_components
  
  y_=tfpl.AutoregressiveTransform(tfb.AutoregressiveNetwork(
        params=2, hidden_units=layer_nf, activation='linear'))(y_)
  
  model=tf.keras.models.Model(inputs=[input_x,input_y], outputs=[y_])
  return model
#%% Function used to make comparison of the target values similar
def forecast_wrapper(array,n_steps_out=7):
    """
    Written by:SJ
    array:predictions;2D vector
    n_steps_out:number of steps out;int
    This function outputs a list with n_steps_out elements containing stepwise forecasts
    returns:list of forecasts
    """
    all_steps=[]
    for ii in range(0,n_steps_out):
        all_=array[ii:len(array):n_steps_out]
        all_steps.append(all_[n_steps_out-1-ii:len(all_)-ii])
    return all_steps
#%%
#forecast function
def model_for_(tf_model,X_test,X_test_era,Y_test,basin_id=0,uq=95,lq=5,LT=7):
  """
  Written by:SJ
  tf_model:trained tf model;tf object
  X_test:scaled input feature;array
  Y_test:scaled target;array
  basin_id:Basin id;int
  uq=upper quantile;int (0-100)
  lq=lower quantile;int (0-100), must be lower than uq otherwise will result in error
  
  The function return an array containing predcitions for median, upper and lower quantiles, and target
  returns:4 X 2D arrays
  """
  id_=100
  predict_all=np.zeros((int(len(X_test)*LT),id_))
  for ii in range(0,id_):
      predict_all[:,ii]=np.asarray(tf_model([X_test,X_test_era]).sample()).ravel()
  #median
  pr_m=np.nanmedian(predict_all,axis=1)
  #upper quantile;
  pr_u=np.nanpercentile(predict_all,q=uq,axis=1)
  #lower quantile
  pr_l=np.nanpercentile(predict_all,q=lq,axis=1)
  
  
  #reading the data
  filename='./ProcessedCamels_AllSame/%s'%(filenames[basin_id])
  # Q is the last column
  df_data=pd.read_csv(filename,header=0)
  #removing date (first column)
  x=np.asarray(df_data.iloc[:,1:])
  x_=np.copy(np.delete(x,13,1))
  train_id,val_id=index_cal(x_)
  x_train_=x_[:train_id]
  mean_=np.mean(x_train_,axis=0)
  sd_=np.std(x_train_,axis=0)
  #mean and sd of Q
  mean_q=mean_[-1]
  sd_q=sd_[-1]
  
  #target
  y_out=np.copy(Y_test)
  
  #step-wise export
  y_out=np.asarray(forecast_wrapper(Y_test.ravel(),LT)).T
  pr_m_out=np.asarray(forecast_wrapper(pr_m.ravel(),LT)).T
  pr_u_out=np.asarray(forecast_wrapper(pr_u.ravel(),LT)).T
  pr_l_out=np.asarray(forecast_wrapper(pr_l.ravel(),LT)).T
  
  ##re-transform
  #median
  pr_m_out=i_normalize(pr_m_out,mean_q,sd_q)
  #uq
  pr_u_out=i_normalize(pr_u_out,mean_q,sd_q)
  #lq
  pr_l_out=i_normalize(pr_l_out,mean_q,sd_q)
  #target
  y_out=i_normalize(y_out,mean_q,sd_q)
  target=y_out[:,0].reshape((-1,1))
  #reshaping to 2D array
  return  pr_l_out,pr_m_out,pr_u_out,target
#%%
#save results function
def model_savedf_(lq_array,m_array,uq_array,target,basin_id=0,uq=95,lq=5):
  """
  Written by:SJ
  lq_array:lower-quantile predictions for all steps;2D array
  m_array:median predictions for all steps;2D array
  uq_array:upper-quantile predictions for all steps;2D array
  target_:observations;2D vector
  basin_id=basin id;int
  This function saves the array outputs in csv format
  returns:-
  """
  #csv column name
  mean_name='./%s/forecast_median_%s'%(save_path_results,filenames[basin_id])
  up_name='./%s/forecast_up_%s'%(save_path_results,filenames[basin_id])
  low_name='./%s/forecast_low_%s'%(save_path_results,filenames[basin_id])
  obs_name='./%s/forecast_obs_%s'%(save_path_results,filenames[basin_id])
  columns_=[]
  for ii in range(1,LT+1):
    columns_.append('LT_%d'%(ii))
  #saving 
  df_mean=pd.DataFrame(m_array,columns=columns_)
  df_mean.to_csv(mean_name,index=None)
  
  df_up=pd.DataFrame(uq_array,columns=columns_)
  df_up.to_csv(up_name,index=None)
  
  df_low=pd.DataFrame(lq_array,columns=columns_)
  df_low.to_csv(low_name,index=None)
  
  df_obs=pd.DataFrame(target,columns=['target'])
  df_obs.to_csv(obs_name,index=None)
  return
#%%
#input lag
IL=365
#lead time
LT=7
#the basins we want to do a regional model for them
list_=list(pd.read_csv('random_numbers_seed_0.csv').iloc[:,1])
for ii in list_:
  print('Catchment #:%d'%(ii+1))
  iid=ii
  
  [x_train,x_val,x_test],[y_train,y_val,y_test]=read_data(basin_id=iid)
  [xx_train,xx_val,xx_test],[yy_train,yy_val,yy_test]=read_data_s(basin_id=iid)
  
  #save the input as a generator
  k=0
  #iterate through catchments
  if k==0:
      with tf.device("CPU"):

          dataset = tf.data.Dataset.from_tensor_slices(((x_train,xx_train), y_train)).shuffle(buffer_size=1024).batch(32)
          dataset_val = tf.data.Dataset.from_tensor_slices(((x_val,xx_val), y_val)).shuffle(buffer_size=1024).batch(32)

  else:
      with tf.device("CPU"):

          dataset_temp = tf.data.Dataset.from_tensor_slices(((x_train,xx_train), y_train)).shuffle(buffer_size=1024).batch(32)
          dataset_val_temp = tf.data.Dataset.from_tensor_slices(((x_val,xx_val), y_val)).shuffle(buffer_size=1024).batch(32)
          
          dataset=dataset.concatenate(dataset_temp)
          dataset_val=dataset_val.concatenate(dataset_val_temp)
          
          del dataset_temp,dataset_val_temp
  k=k+1
  
#initialize the model
    #values selected based on post-processing
dec=model_lstm_(n_features=np.shape(x_train)[-1],lstm_out_dec=200,\
                dense_out_dec=42,num_components=2,component_value=21,\
        activation_dense_dec='linear',nout=LT,nin=365,\
            n_features_era=np.shape(xx_train)[-1])
#use Adam as optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#compile the model
dec.compile(optimizer=optimizer, loss='mse')

#defining the callbacks for early stopping
callback_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=25,
                                                    restore_best_weights=True,
                                                    mode='auto')
callback_learn=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,\
  mode='auto', min_delta=1e-4, cooldown=0, min_lr=1e-6)
st = time.time()

#fit the model
    ##you could change the verbose option
dec.fit(dataset, epochs=100,validation_data=dataset_val, callbacks=[callback_early,callback_learn],batch_size=32,verbose=2)
et = time.time()


#save time
df_time=pd.DataFrame([et-st],columns=['Time (s)'])
name_time='./%s/RunTime_lag_%d_lead_%d_daymet_v1.csv'%(save_path_results,IL,LT)
df_time.to_csv(name_time,index=False)

#save weights
dec_weights='./%s/Weight_lag_%d_lead_%d_RegionalNF_v1.h5'%(iid,IL,LT)
dec.save_weights(dec_weights)


#delete variables to save RAM
del dataset
del dataset_val



for ii in list_:
  print('Catchment #:%d'%(ii+1))
  iid=ii
  
  [x_train,x_val,x_test],[y_train,y_val,y_test]=read_data(basin_id=iid)
  [xx_train,xx_val,xx_test],[yy_train,yy_val,yy_test]=read_data_s(basin_id=iid)
  
  #delete variables to save RAM
  del xx_val
  del yy_val
  del yy_train
  del yy_test
  
  
  
  #use Adam as opimizer
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)
  
  #load model weights for finetuning
  model_=dec.load_weights(dec_weights)
  model_.compile(optimizer=optimizer, loss='mse')
  #finetune the model for 15 epochs
      ##epochs can change
  model_.fit(x=[x_train,xx_train],y=[y_train],epochs=15,verbose=0)
  
  #produce the forecasts
  pr_l_out,pr_m_out,pr_u_out,y_out= model_for_(model_,x_test,xx_test,y_test,basin_id=iid,uq=95,lq=5)
  
  #save results
  model_savedf_(pr_l_out,pr_m_out,pr_u_out,y_out,basin_id=iid,uq=95,lq=5)
  
  #save weights
  model_weights='./%s/Weight_%d_lag_%d_lead_%d_RegionalNFTuned_v1.h5'%(save_path_results,iid,IL,LT)
  model_.save_weights(model_weights)
  
  
  #delete repetetive variables to save RAM
  del model_
  del x_train
  del x_val
  del x_test
  
  del y_train
  del y_val
  del y_test
  
  
  