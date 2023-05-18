# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:21:53 2023

@author: srpv
"""
from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

def filter(signal_window,sample_rate):
    
   
    lowpass = 150000  # Cut-off frequency of the filter
    lowpass_freq = lowpass / (sample_rate / 2) # Normalize the frequency
    b, a = signal.butter(5, lowpass_freq, 'low')
    signal_window = signal.filtfilt(b, a, signal_window)
    
    return signal_window

def Frequencyplot(rawspace,sample_rate,choicewindow,ax):
    
    data_new=rawspace[choicewindow]
    data_new= filter(data_new,sample_rate)
    
    # Define window length (4 seconds)
    win = 0.1 * sample_rate
    freqs, psd = signal.welch(data_new, sample_rate, nperseg=win)
    
    # Plot the power spectrum
    # sns.set(font_scale=1.5, style='white')

    ax.plot(freqs, psd, color='k', lw=0.4)
    sec1, sec2 = 0, 30000
    sec3, sec4 = 30000, 60000
    sec5, sec6 = 60000, 90000
    sec7, sec8 = 90000, 120000
    sec9, sec10 = 120000, 150000
    

# Find intersecting values in frequency vector
    idx_delta1 = np.logical_and(freqs >= sec1, freqs <= sec2)
    idx_delta2 = np.logical_and(freqs >= sec3, freqs <= sec4)
    idx_delta3 = np.logical_and(freqs >= sec5, freqs <= sec6)
    idx_delta4 = np.logical_and(freqs >= sec7, freqs <= sec8)
    idx_delta5 = np.logical_and(freqs >= sec9, freqs <= sec10)
    

    ax.set_ylabel('PSD \n(dB/Hz)',fontsize=15)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlim([0, 150000])
    
    ax.fill_between(freqs, psd, where=idx_delta1, color='#657e39')
    ax.fill_between(freqs, psd, where=idx_delta2, color='#aa332d')
    ax.fill_between(freqs, psd, where=idx_delta3, color='#f0a334')
    ax.fill_between(freqs, psd, where=idx_delta4, color='#0080ff')
    ax.fill_between(freqs, psd, where=idx_delta5, color='#b05aac')
    
    ax.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
    ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    ax.get_yaxis().get_offset_text().set_position((1.15,0))
    
    
def normalize(Features):
    
    df = pd.DataFrame(Features)  
    # df = df.apply(lambda x: (x - np.min(x))/(np.max(x)-np.min(x)), axis=1)
    df = df.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    df = df.to_numpy()
    return df

def Normal_Regime(path,dataset, classname, windowsize):
    
    Normal = classname 
    
    print("dataset_path...",path)
    print("dataset_name...",dataset)
    
    dataset_name=str(dataset)+'_rawspace_5000.npy'
    dataset_label= str(dataset)+'_classspace_5000.npy'
    
    Featurespace = np.load("{}/{}".format(path, dataset_name))
    classspace = np.load("{}/{}".format(path, dataset_label))
    
    Featurespace = pd.DataFrame(Featurespace)
    classspace = pd.DataFrame(classspace)
    classspace.columns = ['Categorical']
    
    
    class_1 = 'LoF pores' 
    class_2 = 'Conduction mode' 
    class_3 = 'Keyhole pores'   
    
    classspace=classspace['Categorical'].replace(0,class_1)
    classspace = pd.DataFrame(classspace)
    classspace=classspace['Categorical'].replace(1,class_2)
    classspace = pd.DataFrame(classspace)
    classspace=classspace['Categorical'].replace(2,class_3)
    classspace = pd.DataFrame(classspace)
    
    data = pd.concat([Featurespace, classspace], axis=1)
    
    print("respective windows",data.Categorical.value_counts())
    minval = min(data.Categorical.value_counts())
    
    print("Before equalizing: ",minval)
    if minval >=2500:
        minval=2500
    else:
        minval=minval
    
    print("windows of the class: ",minval)
    
    data_1 = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique() ])  
    print("The dataset is well balanced: ",data_1.Categorical.value_counts())
    
    data_1 = data_1[data_1.Categorical == str(Normal)]
    
    data=data_1.iloc[:,:-1]
    label=data_1.iloc[:,-1]
    
    x = data.to_numpy() 
    y = label.to_numpy() 
    
    (x, _)=(x[:1], x[1:])
    (y, _)=(y[:1], y[1:])
    
    print("Name of the class...... ",y)
    
    x=normalize(x)
    
    return x
    
    