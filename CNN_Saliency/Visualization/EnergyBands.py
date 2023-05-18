# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:09:53 2023

@author: srpv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from matplotlib import cm
from scipy import signal
import matplotlib.patches as mpatches
from matplotlib import colors
import os

sns.set(font_scale = 1.5)
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_style("ticks", {"xtick.major.size":8,"ytick.major.size":8})

#%%

def boxcomparisonplots(y_pred,y_true,dataset):
    plt.rcParams.update(plt.rcParamsDefault)
    
    folder = os.path.join('Figures/', dataset)
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    
    Featurespace = pd.DataFrame(y_pred)
    classspace = pd.DataFrame(y_true)
    
    classspace.columns = ['Categorical']
                      
    data = pd.concat([Featurespace, classspace], axis=1)
    minval = min(data.Categorical.value_counts())
    print(minval)
    data = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique() ])    
    
    Featurespace=data.iloc[:,:-1]
    classspace=data.iloc[:,-1]
    values, counts = np.unique(classspace, return_counts=True)
    print(values,counts)
    
    classspace = classspace.to_numpy() 
    Featurespace = Featurespace.to_numpy() 
    
    c=len(Featurespace)
    df1 = pd.DataFrame(Featurespace)
    df1 = np.ravel(df1,order='F')
    df1=pd.DataFrame(df1)
    
    df2 = pd.DataFrame(classspace)
    df2.columns = ['Categorical']
    

    a=df2.stack().value_counts() 
    
    
    filename = 'Dec-1'
    numbers = np.random.randn(c)
    df3 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df3=df3.drop(['numbers'], axis=1)
	
    filename = 'Dec-2'
    numbers = np.random.randn(c)
    df4 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df4=df4.drop(['numbers'], axis=1)
	
    filename = 'Dec-3'
    numbers = np.random.randn(c)
    df5 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df5=df5.drop(['numbers'], axis=1)
	
    filename = 'Dec-4'
    numbers = np.random.randn(c)
    df6 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df6=df6.drop(['numbers'], axis=1)
    
    filename = 'Dec-5'
    numbers = np.random.randn(c)
    df7 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df7=df7.drop(['numbers'], axis=1)
    
    filename = 'Dec-6'
    numbers = np.random.randn(c)
    df8 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df8=df8.drop(['numbers'], axis=1)
    
    filename = 'Dec-7'
    numbers = np.random.randn(c)
    df9 = pd.DataFrame({'labels': filename , 'numbers': numbers})
    df9=df9.drop(['numbers'], axis=1)
	
   
    Energyband=np.concatenate((df3,df4,df5,df6,df7,df8,df9), axis=0)  
    Modes=np.concatenate((df2,df2,df2,df2,df2,df2,df2), axis=0)
    
    Energybands = np.concatenate((df1,Energyband,Modes), axis=1)    
    Energybands = pd.DataFrame(Energybands)
    Energybands.columns = ['Importance score','Decompositions levels','Categorical']
    
    

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(7, 5))
    sns.set(font_scale = 4.5)
    sns.set_style("whitegrid", {'axes.grid' : False})
    ax=sns.catplot(y="Decompositions levels",x="Importance score", hue="Categorical", kind="bar", data=Energybands, height=12, 
                    aspect=1.8, palette={0: "blue", 1: "green",2: "red"})
    ax.set_xticklabels(rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=50)
    ax.tick_params(axis='both', which='minor', labelsize=50)
    ax._legend.remove()
    # plt.legend(loc='upper right',frameon=False,fontsize=30)
    # plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    plt.title('Dataset '+str(dataset), fontsize=50)
    plotname=folder+ str(dataset)+'_'+"Decomposition"+'_Importance.png'
    plt.savefig(plotname, dpi=800,bbox_inches='tight')
    plt.show()
    

