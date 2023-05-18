# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:22:57 2023

@author: srpv
"""



import numpy as np
import pandas as pd
from Utils import *


'''
Download the dataset from following link
-- https://polybox.ethz.ch/index.php/s/9xLsB1kJORolfTc
'''

sample_rate = 400000
windowsize= 5000
N=windowsize
t0=0
dt=1/sample_rate
time = np.arange(0, N) * dt + t0


#%%
path=r'C:\Users\srpv\Desktop\LPBF Sensor selection Decomposition\Data' #Data folder path
sensor=['D1','D2','D3','D4']

for k in sensor:
    
    print(k)
    Featurespace,label= decompositions(path,k,num_imf=7,sample_rate=sample_rate)
    data_filename= str(k)+'_IMF_7_Decompositions'+'.npy'
    np.save(data_filename,Featurespace, allow_pickle=True)
    data_filename= str(k)+'_Classspace_IMF_7_Decompositions'+'.npy'
    np.save(data_filename,label, allow_pickle=True)