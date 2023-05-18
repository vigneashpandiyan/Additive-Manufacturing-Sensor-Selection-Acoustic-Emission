# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:21:53 2023

@author: srpv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *


sns.set(font_scale = 1.5)
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_style("ticks", {"xtick.major.size":8,"ytick.major.size":8})
#%%

sample_rate = 400000
windowsize= 5000
N=windowsize
t0=0
dt=1/sample_rate
time = np.arange(0, N) * dt + t0
sensor=['D1','D2','D3','D4']


for k in sensor:
    
    featurefile_1 = str(k)+'_'+'featurespace_5000_FFT'+'.npy'
    classfile_1 = str(k)+'_Classspace_5000_FFT'+'.npy'

    Featurespace = np.load(featurefile_1).astype(np.float64)
    classspace= np.load(classfile_1).astype(np.float64)

    Featurespace = pd.DataFrame(Featurespace)
    classspace = pd.DataFrame(classspace)
    classspace.columns = ['Categorical']
                      
    data = pd.concat([Featurespace, classspace], axis=1)
    minval = min(data.Categorical.value_counts())

    data = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique() ])    

    Featurespace=data.iloc[:,:-1]
    classspace=data.iloc[:,-1]

    values, counts = np.unique(classspace, return_counts=True)
    
    print(values,counts)
    
    classspace = classspace.to_numpy() 
    Featurespace = Featurespace.to_numpy()

    Energybands=boxcomparisonplots(Featurespace,classspace)
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(12, 8))
    sns.set(font_scale = 3.5)
    sns.set_style("whitegrid", {'axes.grid' : False})
    ax=sns.catplot(x="Frequency Bands",y="Cumulative energy value", hue="Categorical", kind="bar", data=Energybands, height=9, 
                    aspect=2.8,palette={"LoF pores": "blue", "Conduction mode": "green","Keyhole pores": "red"})
    
    ax.set_xticklabels(rotation=45,fontsize=45)
    ax.tick_params(axis='both', which='major', labelsize=45)
    ax.tick_params(axis='both', which='minor', labelsize=45)
    ax._legend.remove()
    plt.ylabel("Cumulative energy value \n (dB/Hz)",labelpad=20)
    plt.title('Dataset '+ k,fontsize=60)
    # plt.legend(loc='upper right',frameon=False,fontsize=60)
    plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    plotname=str(k)+'_'+"kHz 0-150"+'_Energyband_Comparison_FFT.png'
    plt.savefig(plotname, dpi=800,bbox_inches='tight')
    plt.show()