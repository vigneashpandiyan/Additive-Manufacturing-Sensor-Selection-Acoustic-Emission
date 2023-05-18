# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:37:13 2022

@author: srpv
"""
import numpy as np
import pandas as pd
import torch

from CNN_Saliency.Network.Model import CNN,LinearNetwork
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluation(test_loader_lineval, nb_class, ckpt,lkpt, opt, folder):
    
    folder = os.path.join('Figures/', folder)
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    # no augmentations used for linear evaluation
    
    backbone_lineval = CNN(opt.feature_size).cuda()  # defining a raw backbone model
    checkpoint = torch.load(ckpt, map_location='cpu')
    backbone_lineval.load_state_dict(checkpoint)
    
    print(opt.feature_size)
    
    linear_layer = LinearNetwork(opt.feature_size, len(nb_class)).cuda()
    checkpoint = torch.load(lkpt, map_location='cpu')
    linear_layer.load_state_dict(checkpoint)
    
      
    backbone_lineval.eval()    
    linear_layer.eval()
    
    with torch.no_grad():
        y_pred = []
        y_true = []
        for data, target in test_loader_lineval:
            
            
            data = data.cuda()
            target = target.cuda()

            output = backbone_lineval(data).detach()
            output = linear_layer(output)
            # estimate the accuracy
            prediction = output.argmax(-1)
            prediction=prediction.data.cpu().numpy()
            output=target.data.cpu().numpy()
            
            y_true.extend(output) # Save Truth 
            y_pred.extend(prediction) # Save Prediction
            
        plot_confusion_matrix(y_true, y_pred,folder)

def plot_confusion_matrix(y_true, y_pred,folder):
    plt.rcParams.update(plt.rcParamsDefault)
    classes = ('1', '2', '3')
    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalise
    cmn = cm.astype('float')  / cm.sum(axis=1)[:, np.newaxis]
    cmn=cmn*100
    
    fig, ax = plt.subplots(figsize=(12,9))
    sns.set(font_scale=3) 
    b=sns.heatmap(cmn, annot=True, fmt='.1f', xticklabels=classes, yticklabels=classes,cmap="coolwarm",linewidths=0.1,annot_kws={"size": 25},cbar_kws={'label': 'Classification Accuracy %'})
    for b in ax.texts: b.set_text(b.get_text() + " %")
    plt.ylabel('Actual',fontsize=25)
    plt.xlabel('Predicted',fontsize=25)
    plt.margins(0.2)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center", fontsize= 20)
    ax.set_xticklabels(ax.get_xticklabels(), va="center",fontsize= 20)
    # plt.setp(ax.get_yticklabels(), rotation='vertical')
    plotname=folder+'Classification_accuracy.png'
    plotname=str(plotname)
    plt.savefig(plotname,bbox_inches='tight')
    plt.show()
    plt.close()


def soft_max(v):
    v2 = (v + min(v)) / sum(v + min(v))
    return v2


def saliency(train, nb_class,ckpt,lkpt, opt):
    #we don't need gradients w.r.t. weights for a trained model
    
    y_pred = []
    y_true = []
    
    train_loader_lineval = torch.utils.data.DataLoader(train, batch_size=1, num_workers=0,
                                                    shuffle=False)
    # loading the saved backbone
    backbone_lineval = CNN().cuda()  # defining a raw backbone model
    checkpoint = torch.load(ckpt, map_location='cpu')
    backbone_lineval.load_state_dict(checkpoint)
   
    for param in backbone_lineval.parameters():
       param.requires_grad = False 
   
    linear_layer = LinearNetwork(opt.feature_size, len(nb_class)).cuda()
    checkpoint = torch.load(lkpt, map_location='cpu')
    linear_layer.load_state_dict(checkpoint)
    
    for param_1 in linear_layer.parameters():
        param_1.requires_grad = False
    
    backbone_lineval.eval()    
    linear_layer.eval()
    
    
    for data, target in train_loader_lineval:
    #transoform input PIL image to torch.Tensor and normalize
        
        input = data.cuda()
        input.requires_grad = True
        output = backbone_lineval(input)
        preds = linear_layer(output)
        score, indices = torch.max(preds, 1)
        target=target.detach().cpu().numpy()
        indices=indices.detach().cpu().numpy()
       
        if target == indices:
        
            score.backward()
            slc, _ = torch.max(torch.abs(input.grad), dim=0)
            slc = (slc - slc.min())/(slc.max()-slc.min())
            slc=slc.detach().cpu().numpy()
            slc= np.sum(slc, axis = 1)
            slc=soft_max(slc)
            y_true.extend(target) # Save Truth 
            y_pred.append(slc) 
    
    y_pred=np.array(y_pred)

    return y_true,y_pred



def saliency_class(train, nb_class,ckpt,lkpt, opt,dataset,label,color):
    #we don't need gradients w.r.t. weights for a trained model
    
    plt.rcParams.update(plt.rcParamsDefault)
    folder = os.path.join('Figures/', dataset)
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    
    y_pred = []
    y_true = []
    
    train_loader_lineval = torch.utils.data.DataLoader(train, batch_size=1, num_workers=0,
                                                    shuffle=False)
    # loading the saved backbone
    backbone_lineval = CNN().cuda()  # defining a raw backbone model
    checkpoint = torch.load(ckpt, map_location='cpu')
    backbone_lineval.load_state_dict(checkpoint)
   
    for param in backbone_lineval.parameters():
       param.requires_grad = False 
   
    linear_layer = LinearNetwork(opt.feature_size, len(nb_class)).cuda()
    checkpoint = torch.load(lkpt, map_location='cpu')
    linear_layer.load_state_dict(checkpoint)
    
    for param_1 in linear_layer.parameters():
        param_1.requires_grad = False
    
    backbone_lineval.eval()    
    linear_layer.eval()
    
    
    for data, target in train_loader_lineval:
    #transoform input PIL image to torch.Tensor and normalize
        
        input = data.cuda()
        input.requires_grad = True
        output = backbone_lineval(input)
        preds = linear_layer(output)
        score, indices = torch.max(preds, 1)
        target=target.detach().cpu().numpy()
        indices=indices.detach().cpu().numpy()
        
        
        if label==target:
       
            if target == indices:
                # print(target)
                score.backward()
                slc, _ = torch.max(torch.abs(input.grad), dim=0)
                slc = (slc - slc.min())/(slc.max()-slc.min())
                slc=slc.detach().cpu().numpy()
                slc= np.sum(slc, axis = 1)
                slc=soft_max(slc)
                y_true.extend(target) # Save Truth 
                y_pred.append(slc) 
    
    y_pred=np.array(y_pred)
    
    plt.rcParams.update(plt.rcParamsDefault)
    df = pd.DataFrame(y_pred)
    df.columns =['Dec-1', 'Dec-2', 'Dec-3', 'Dec-4', 'Dec-5', 'Dec-6', 'Dec-7']
    fig, ax = plt.subplots(figsize=(5,3))
    sns.set_theme(style='white')
    sns.set_style('ticks')
    sns.despine()
    sns.barplot(data=df,color=color,ax=ax, orient='h')
    plt.xlabel('Importance score (Normalized)') 
    plt.ylabel('IMF Levels') 
    title=folder+str(dataset)+'_Importance'+'_'+str(label)+'_'+'plot'+'.png'
    plt.savefig(title, bbox_inches='tight',dpi=400)
    plt.show()

    return y_true,y_pred




