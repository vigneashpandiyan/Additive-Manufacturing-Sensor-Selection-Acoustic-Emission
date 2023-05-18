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

def training_curves(Training_loss,Training_accuracy,learn_rate,folder):
    
    Training_loss=np.asarray(Training_loss)
    Training_loss=Training_loss.astype(np.float64)
    classfile = folder+'Training_loss'+'.npy'
    np.save(classfile,Training_loss, allow_pickle=True)
    
    
    Training_accuracy=np.asarray(Training_accuracy)
    Training_accuracy=Training_accuracy.astype(np.float64)
    classfile = folder+'Training_accuracy'+'.npy'
    np.save(classfile,Training_accuracy, allow_pickle=True)
    
    
    learning_rate=np.asarray(learn_rate)
    learning_rate=learning_rate.astype(np.float64)
    classfile = folder+'learning_rate'+'.npy'
    np.save(classfile,learning_rate, allow_pickle=True)
    
    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize = (5, 3))
    plt.plot(Training_loss,'blue') 
    plt.title('Training loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(labels=['Training loss'], loc='best',fontsize=10,frameon=False)
    plotname=folder+'Training_loss'+'.png'
    plt.savefig(plotname, bbox_inches='tight',dpi=200)
    plt.show()
    
    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize = (5, 3))
    plt.plot(Training_accuracy,'red') 
    plt.title('Training accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(labels=['Training accuracy'], loc='best',fontsize=10,frameon=False)
    plotname=folder+'Training_accuracy'+'.png'
    plt.savefig(plotname, bbox_inches='tight',dpi=200)
    plt.show()
    
    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize = (5, 3))
    plt.plot(learning_rate,'black') 
    plt.title('Learning rate')
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend(labels=['Learning rate'], loc='best',fontsize=10,frameon=False)
    plotname=folder+'learning_rate'+'.png'
    plt.savefig(plotname, bbox_inches='tight',dpi=200)
    plt.show()
    
   