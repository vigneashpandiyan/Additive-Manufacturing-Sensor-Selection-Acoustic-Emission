# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:29:36 2023

@author: srpv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os     
import itertools
from Classifiers.XGBoost import *
from Classifiers.Helper import *
from Classifiers.plot_roc import *

from sklearn.model_selection import train_test_split# implementing train-test-split

#%%

def classifier_ML(path, dataset_name):
    
    print("dataset_path...",path)
    print("dataset_name...",dataset_name)
    
    raw=str(dataset_name)+'_featurespace_5000_FFT'+ '.npy'
    label=str(dataset_name)+'_Classspace_5000_FFT'+'.npy'
    
    Featurespace = np.load("{}/{}".format(path,raw))
    classspace = np.load("{}/{}".format(path,label))
    
    # Featurespace=Featurespace.to_numpy()
    Featurespace=(Featurespace[:, 0:15]) #44:59
    folder=dataset_name +'_'+'XG_Boost_ML'
    
    folder = os.path.join('Figures/', folder)
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    
    classspace = pd.DataFrame(classspace)
    Featurespace = pd.DataFrame(Featurespace).astype(np.float64)
    
    num_cols = len(list(Featurespace))
    rng = range(1, num_cols + 1)
    # Featurenames = ['Feature_' + str(i) for i in rng] 
    
    Featurenames = [str((i-1)*10)+'-' +str((i)*10)+ ' kHz' for i in rng] 
    Featurespace.columns = Featurenames
    feature_cols=list(Featurespace.columns) 
    Featurespace.info()
    Featurespace.describe()
    Featurespace.head()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(Featurespace, classspace, test_size=0.25, random_state=66)

    classes=np.unique(classspace)
    classes = list(classes)
    
    XGBoost(X_train, X_test, y_train, y_test,Featurespace, classspace,classes,folder,dataset_name)
   




path=r'C:\Users\srpv\Desktop\LPBF Sensor selection Decomposition\Feature extraction FFT' #Data folder path
sensor=['D1','D2','D3','D4']

for k in sensor:

    classifier_ML(path, k)
    
   
