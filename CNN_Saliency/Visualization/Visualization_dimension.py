# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:37:13 2022

@author: srpv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import metrics
import seaborn as sns

def distribution_plot(data,i,folder):
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)
    
    data_1 = data[data.target == 'LoF']
    data_1 = data_1.drop(labels='target', axis=1)
    data_2 = data[data.target == 'Conduction mode']
    data_2 = data_2.drop(labels='target', axis=1)
    data_3 = data[data.target == 'Keyhole']
    data_3 = data_3.drop(labels='target', axis=1)
    
    Sensor="Flat"
    
    sns.set(style="white")
    fig=plt.subplots(figsize=(5,3), dpi=800)
    fig = sns.kdeplot(data_1['Feature'], shade=True,alpha=.5, color="#0000FF")
    fig = sns.kdeplot(data_2['Feature'], shade=True,alpha=.5, color="green")
    fig = sns.kdeplot(data_3['Feature'], shade=True,alpha=.5, color="red")
    
    
    data=pd.concat([data_1,data_2,data_3],axis=1) 
    data=data.to_numpy()
    
    plt.title("Weight " + str(i+1))
    plt.legend(labels=['LoF pores','Conduction mode','Keyhole pores'],bbox_to_anchor=(1.49, 1.05))
    title=folder+'Dimension'+'_'+str(i+1)+'_'+'distribution_plot'+'.png'
    # plt.xlim([0.0, np.max(data)])
    # plt.ylim([0.0, 65])
    plt.xlabel('Weight distribution') 
    plt.savefig(title, bbox_inches='tight')
    plt.show()
    
#%%

def plots(folder):
    
    folder = os.path.join('Figures/', folder)
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    
    train_embeddings = folder+ 'embeddings'+ '.npy'
    train_labelsname = folder+  'labels'+'.npy'

    X_train = np.load(train_embeddings).astype(np.float64)
    y_train = np.load(train_labelsname).astype(np.float64)
    
    Featurespace = X_train
    classspace= y_train 

    columns = np.atleast_2d(Featurespace).shape[1]
    df2 = pd.DataFrame(classspace)
    
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'LoF')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'Conduction mode')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'Keyhole')
    df2 = pd.DataFrame(df2)
    print(columns)
    
    for i in range(columns):
        print(i)
        Featurespace_1 = Featurespace.transpose()
        data=(Featurespace_1[i])
        data=data.astype(np.float64)
        #data= abs(data)
        df1 = pd.DataFrame(data)
        df1.rename(columns={df1.columns[0]: "Feature" }, inplace = True)
        df2.rename(columns={df2.columns[0]: "categorical" }, inplace = True)
        data = pd.concat([df1, df2], axis=1)
        minval = min(data.categorical.value_counts())
        data = pd.concat([data[data.categorical == cat].head(minval) for cat in data.categorical.unique() ])
        distribution_plot(data,i,folder)
     
    all_plot(X_train,y_train,folder)

def Cummulative_plots(Featurespace,classspace,i,ax):
    
    columns = np.atleast_2d(Featurespace).shape[1]
    df2 = pd.DataFrame(classspace)
    
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'LoF')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'Conduction mode')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'Keyhole')
    df2 = pd.DataFrame(df2)
    
    
    print(i)
    
    Featurespace_1 = Featurespace.transpose()
    data=(Featurespace_1[i])
    data=data.astype(np.float64)
    
    df1 = pd.DataFrame(data)
    df1.rename(columns={df1.columns[0]: "Feature" }, inplace = True)
    df2.rename(columns={df2.columns[0]: "categorical" }, inplace = True)
    data = pd.concat([df1, df2], axis=1)
    minval = min(data.categorical.value_counts())
    data = pd.concat([data[data.categorical == cat].head(minval) for cat in data.categorical.unique() ])
    
    Cummulative_dist_plot(data,i,ax)
    
def Cummulative_dist_plot(data,i,ax):
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)
    
    data_1 = data[data.target == 'LoF']
    data_1 = data_1.drop(labels='target', axis=1)
    data_2 = data[data.target == 'Conduction mode']
    data_2 = data_2.drop(labels='target', axis=1)
    data_3 = data[data.target == 'Keyhole']
    data_3 = data_3.drop(labels='target', axis=1)
    
    plt.rcParams.update(plt.rcParamsDefault)
    sns.set(style="white")
    
    ax.plot(figsize=(5,5), dpi=800)
    sns.kdeplot(data_1['Feature'], shade=True,alpha=.5, color="blue",ax=ax)
    sns.kdeplot(data_2['Feature'], shade=True,alpha=.5, color="green",ax=ax)
    sns.kdeplot(data_3['Feature'], shade=True,alpha=.5, color="red",ax=ax)
    
    
    ax.set_title("Weight " + str(i+1), y=1.0, pad=-14)
    ax.set_xlabel('Weight distribution') 
    # ax.set_ylabel('Density')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    

    
def all_plot(X_train,y_train,folder):
    
    fig, axs = plt.subplots(
      nrows=8,
      ncols=8,
      sharey=False,
      figsize=(20, 20),
      dpi=600
    )
    
    columns = np.atleast_2d(X_train).shape[1]
    graph_name=folder+'Byol_Latent_'+str(columns)+'D_'+'.png'
    
    for i in range(columns):
      ax = axs.flat[i]
      Cummulative_plots(X_train,y_train,i,ax)
      
    fig.tight_layout();
    fig.savefig(graph_name)
    fig.show()

