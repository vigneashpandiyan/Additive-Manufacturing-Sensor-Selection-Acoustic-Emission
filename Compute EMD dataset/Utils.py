# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:43:13 2023

@author: srpv
"""

import torch
import numpy as np
from matplotlib import pyplot as plt
from EMD import ComputeEMD
from scipy.signal import chirp
import scipy.signal as signal
import IPython
import pandas as pd


#%%

def normalize(Features):
    
    df = pd.DataFrame(Features)  
    df = df.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    df = df.to_numpy()
    return df

def load_LPBF(path, dataset_name,dataset_label):
    ##################
    # load raw data
    ##################
    
    print("dataset_path...",path)
    print("dataset_name...",dataset_name)
    
    Featurespace = np.load("{}/{}".format(path, dataset_name))
    classspace = np.load("{}/{}".format(path, dataset_label))
    
    Featurespace = pd.DataFrame(Featurespace)
    classspace = pd.DataFrame(classspace)
    classspace.columns = ['Categorical']
    data = pd.concat([Featurespace, classspace], axis=1)
    
    
    minval = min(data.Categorical.value_counts())
    
    if minval >=3000:
        minval=3000
    else:
        minval=minval
    
    print("windows of the class: ",minval)
    
    minval=10
    data_1 = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique() ])  
    
    print("The dataset is well balanced: ",data_1.Categorical.value_counts())
    
    data=data_1.iloc[:,:-1]
    label=data_1.iloc[:,-1]
    
    x=normalize(data)
    y = label.to_numpy() 
    
   
    return x , y


def Computedecompositions(val,num_imf,sample_rate):
    

    imfs = ComputeEMD(val, sample_rate, num_imf=num_imf)
    imfs=imfs.detach().cpu().numpy().squeeze()
    
    Feature_vectors=np.empty((0, imfs.shape[0],imfs.shape[1]))
    Feature_vectors = np.append(Feature_vectors,[imfs], axis=0)
    return Feature_vectors

def decompositions(path,sensor,num_imf,sample_rate):
    
    dataset_name=str(sensor)+'_rawspace_5000.npy'
    dataset_label= str(sensor)+'_classspace_5000.npy'
    data_new, label= load_LPBF(path, dataset_name,dataset_label)
    
    iteration = np.atleast_2d(data_new).shape[0]
    print("iteration to be done......",iteration)
    featurelist=[]
    
    #for row in loop:
    for k in range(iteration):
        val= data_new[k,:]
        val = np.expand_dims(val, axis=0)
        Feature_vectors=Computedecompositions(val,num_imf,sample_rate)
        
        print(k)
        
        for item in Feature_vectors:
            featurelist.append(item)
    return featurelist,label

