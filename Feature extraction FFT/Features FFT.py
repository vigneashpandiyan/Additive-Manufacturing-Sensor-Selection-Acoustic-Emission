# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:00:09 2023

@author: srpv
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
from utils import *

print(np.__version__)

'''
Download the dataset from following link
-- https://polybox.ethz.ch/index.php/s/9xLsB1kJORolfTc
'''

#%%
sample_rate = 400000
windowsize= 5000
N=windowsize
t0=0
dt=1/sample_rate
time = np.arange(0, N) * dt + t0


#%%



path=r'C:\Users\srpv\Desktop\LPBF Sensor selection Decomposition\Data' #Data folder path
sensor=['D1','D2','D3','D4']
band_size = 16 

for k in sensor:
    

    dataset_name=str(k)+'_rawspace_5000.npy'
    dataset_label= str(k)+'_classspace_5000.npy'

    data_new, label= load_LPBF(path, dataset_name,dataset_label)

    featurelist= spectrum_FFT(data_new,sample_rate,band_size)
    Featurespace=np.asarray(featurelist)
    Featurespace=Featurespace.astype(np.float64)
    
    data_filename= str(k)+'_featurespace_5000_FFT'+'.npy'
    np.save(data_filename,Featurespace, allow_pickle=True)
    
    data_filename= str(k)+'_Classspace_5000_FFT'+'.npy'
    np.save(data_filename,label, allow_pickle=True)



