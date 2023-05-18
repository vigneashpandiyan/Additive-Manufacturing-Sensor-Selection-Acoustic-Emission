# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 00:24:57 2023

This repo hosts the codes that were used in journal work: 
"Sensor selection for process monitoring based on deciphering acoustic emissions 
from different dynamics of the Laser Powder Bed Fusion process 
using Empirical Mode Decompositions and Interpretable Machine Learning"

--> Cite the following work

"Sensor selection for process monitoring based on deciphering acoustic emissions 
from different dynamics of the Laser Powder Bed Fusion process 
using Empirical Mode Decompositions and Interpretable Machine Learning"

@contact --> vigneashwara.solairajapandiyan@empa.ch

@author: srpv

"""

import torch
import pandas as pd
import numpy as np

from CNN_Saliency.Dataloader.Load_LPBF import data_tensor
from CNN_Saliency.Dataloader.Parser import parse_option


from CNN_Saliency.Trainer.train import supervised_train,count_parameters
from CNN_Saliency.Evaluation.Supervised_eval import evaluation,saliency,saliency_class

from CNN_Saliency.Visualization.TSNE_visualization import dimensionality_reduction
from CNN_Saliency.Visualization.Visualization_dimension import *
from CNN_Saliency.Visualization.EnergyBands import *

from sklearn.model_selection import train_test_split# implementing train-test-split

#%%
'''
Download the dataset from following link
-- https://polybox.ethz.ch/index.php/s/9xLsB1kJORolfTc
'''

path=r'C:\Users\srpv\Desktop\LPBF Sensor selection Decomposition\Compute EMD dataset'
sensor=['D1','D2','D3','D4']

for dataset in sensor:
    
    dataset_name=path+'/'+str(dataset)+'_IMF_7_Decompositions.npy'
    dataset_label=path+'/'+ str(dataset)+'_Classspace_IMF_7_Decompositions.npy'
    folder=dataset
    
    folder = os.path.join('Figures/', folder)
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    
    opt = parse_option()
    
    Featurespace=np.load(dataset_name)
    classspace=np.load(dataset_label) 
    nb_class=[0., 1., 2.]
    
    
    sequences =[]
    
    for i in range(len(classspace)):
        # print(i)
        sequence_features = Featurespace[i]
        label = classspace[i]
        sequences.append((sequence_features,label))
    
    sequences = data_tensor(sequences)
    
    train, test = train_test_split(sequences, test_size=0.2)
    trainset = torch.utils.data.DataLoader(train, batch_size=128, num_workers=0,
                                                    shuffle=True)
    
    testset = torch.utils.data.DataLoader(test, batch_size=128, num_workers=0,
                                                    shuffle=True)
    
    backbone_lineval, linear_layer = supervised_train(trainset, testset, testset, nb_class, opt, folder=dataset)
    
    count_parameters(backbone_lineval)
    count_parameters(linear_layer)
        
    
    folder_created = os.path.join('Figures/', dataset)
    ckpt ='{}/backbone_best.tar'.format(folder_created)
    lkpt = '{}/linear_layer_last.tar'.format(folder_created)
    
    evaluation(testset,nb_class,ckpt,lkpt, opt, folder=dataset)
    X, y,ax,fig,graph_name= dimensionality_reduction(trainset, testset, testset, ckpt,folder=dataset)
    plots(dataset)
    
    y_true,y_pred=saliency(train, nb_class,ckpt,lkpt, opt)
    
    boxcomparisonplots(y_pred,y_true,dataset)
    
    y_true,saliencyvalues=saliency_class(train, nb_class,ckpt,lkpt, opt,dataset,0,'blue')
    y_true,saliencyvalues=saliency_class(train, nb_class,ckpt,lkpt, opt,dataset,1,'green')
    y_true,saliencyvalues=saliency_class(train, nb_class,ckpt,lkpt, opt,dataset,2,'red')


