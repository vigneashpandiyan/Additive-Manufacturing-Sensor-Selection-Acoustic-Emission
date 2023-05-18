# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:21:53 2023

@author: srpv
"""
from scipy import signal
import numpy as np
import pandas as pd
from scipy.signal import welch, periodogram

def filter(signal_window,sample_rate):
    
    lowpass = 150000  # Cut-off frequency of the filter
    lowpass_freq = lowpass / (sample_rate / 2) # Normalize the frequency
    b, a = signal.butter(5, lowpass_freq, 'low')
    signal_window = signal.filtfilt(b, a, signal_window)
    return signal_window


def normalize(Features):

    df = pd.DataFrame(Features)  
    df = df.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    df = df.to_numpy()
    return df


def get_band(band_size,band_max_size):
    band_window = 0
    band = []
    for y in range(band_size):
        band.append(band_window)
        band_window += band_max_size / band_size
    return band

def spectrumpower(psd, band,freqs,band_size):
    length = len(band)
    # print(length)
    Feature_deltapower=[]
    Feature_relativepower=[]
    for i in range(band_size-1):
        if i <= (len(band)): 
            
            ii=i
            low = band[ii]
            
            ii=i+1;
            high = band[ii]
           
            idx_delta = np.logical_and(freqs >= low, freqs <= high)
            total_power = sum(psd)
            delta_power = sum(psd[idx_delta])
            delta_rel_power = delta_power / total_power
            Feature_deltapower.append(delta_power)
            Feature_relativepower.append(delta_rel_power)
                        
    return Feature_deltapower,Feature_relativepower


def function(val,sample_rate,band_size):
    
    i=0
    
    signal_window=filter(val,sample_rate)

    win = 4 * sample_rate
    freqs, psd = periodogram(signal_window, sample_rate,window='hamming')
    band_max_size = 160000
    band = get_band(band_size,band_max_size)
    
    
    Feature1,Feature2 =spectrumpower(psd,band,freqs,band_size)
    Feature1 = np.asarray(Feature1)

    Feature2 = np.asarray(Feature2)
    
    Feature=np.concatenate((Feature1,Feature2))
    
    if  i ==0:
    #     print("--reached")
        size_of_Feature_vectors = int(len(Feature))
        size_of_dataset = int(len(signal_window))
        
        Feature_vectors=np.empty((0, size_of_Feature_vectors))
        rawdataset=np.empty((0, size_of_dataset))
           
   
    #print(label) 
    Feature_vectors = np.append(Feature_vectors,[Feature], axis=0)
    rawdataset = np.append(rawdataset,[signal_window],axis=0)
              
    return Feature_vectors
            
#%%

def spectrum_FFT(data_new,sample_rate,band_size):
    columnsdata=data_new.transpose()
    columns = np.atleast_2d(columnsdata).shape[1]
    featurelist=[]
    classlist=[]
    rawlist =[]
    
    #for row in loop:
    for k in range(columns):
        
        val= columnsdata[:,k]
        Feature_vectors=function(val,sample_rate,band_size)
        
        print(k)
        
        for item in Feature_vectors:
            
            featurelist.append(item)
        
    return featurelist




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
    
    minval=5
    data_1 = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique() ])  
    data=data_1.iloc[:,:-1]
    label=data_1.iloc[:,-1]
    
    x=normalize(data)
    
    y = label.to_numpy() 
    
    return x , y


def boxcomparisonplots(Featurespace,classspace):
    
    Material_1=(Featurespace[:, 0:15]).astype(np.float64) #44:59
    c=len(Material_1)
    df1 = pd.DataFrame(Material_1)
    df1 = np.ravel(df1,order='F')
    df1=pd.DataFrame(df1)
    
    
    df2 = pd.DataFrame(classspace)
    df2.columns = ['Categorical']
    
    df2=df2['Categorical'].replace(0,'LoF pores')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'Conduction mode')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'Keyhole pores')
    df2 = pd.DataFrame(df2)
    
    
    
    a=df2.stack().value_counts() 
    
    result1 = a.get(key = 'LoF pores') 
    result2 = a.get(key = 'Conduction mode') 
    result3 = a.get(key = 'Keyhole pores') 
    
    
    
        
    filename = '0-10 kHz'
    numbers = np.random.randn(c)
    df3 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df3=df3.drop(['numbers'], axis=1)
	
    filename = '10-20 kHz'
    numbers = np.random.randn(c)
    df4 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df4=df4.drop(['numbers'], axis=1)
	
    filename = '20-30 kHz'
    numbers = np.random.randn(c)
    df5 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df5=df5.drop(['numbers'], axis=1)
	
    filename = '30-40 kHz'
    numbers = np.random.randn(c)
    df6 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df6=df6.drop(['numbers'], axis=1)
	
    filename = '40-50 kHz'
    numbers = np.random.randn(c)
    df7 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df7=df7.drop(['numbers'], axis=1)
	
    filename = '50-60 kHz'
    numbers = np.random.randn(c)
    df8 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df8=df8.drop(['numbers'], axis=1)
	
    filename = '60-70 kHz'
    numbers = np.random.randn(c)
    df9 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df9=df9.drop(['numbers'], axis=1)
	
    filename = '70-80 kHz'
    numbers = np.random.randn(c)
    df10 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df10=df10.drop(['numbers'], axis=1)
    
    filename = '80-90 kHz'
    numbers = np.random.randn(c)
    df11 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df11=df11.drop(['numbers'], axis=1)
    
    filename = '90-100 kHz'
    numbers = np.random.randn(c)
    df12 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df12=df12.drop(['numbers'], axis=1)
    
    filename = '100-110 kHz'
    numbers = np.random.randn(c)
    df13 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df13=df13.drop(['numbers'], axis=1)
    
    filename = '110-120 kHz'
    numbers = np.random.randn(c)
    df14 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df14=df14.drop(['numbers'], axis=1)
    
    filename = '120-130 kHz'
    numbers = np.random.randn(c)
    df15 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df15=df15.drop(['numbers'], axis=1)
	
    filename = '130-140 kHz'
    numbers = np.random.randn(c)
    df16 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df16=df16.drop(['numbers'], axis=1)
	
    filename = '140-150 kHz'
    numbers = np.random.randn(c)
    df17 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df17=df17.drop(['numbers'], axis=1)
	
    
    Energyband=np.concatenate((df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17), axis=0)  
    Modes=np.concatenate((df2,df2,df2,df2,df2,df2,df2,df2,df2,df2,df2,df2,df2,df2,df2), axis=0)
    
    Energybands = np.concatenate((df1,Energyband,Modes), axis=1)    
    Energybands = pd.DataFrame(Energybands)
    
    Energybands.columns = ['Cumulative energy value','Frequency Bands','Categorical']
    #Inconel.columns = ['20','Categorical']
    return Energybands

