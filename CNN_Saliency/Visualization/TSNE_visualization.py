# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:47:42 2022
@author: srpv


"""

from matplotlib import animation
import torch
import torch.utils.data as data
from CNN_Saliency.Trainer.pytorchtools import EarlyStopping
from CNN_Saliency.Network.Model import CNN
from CNN_Saliency.Visualization.TSNE import *
import os
#%%

def dimensionality_reduction(train_loader_lineval, val_loader_lineval, test_loader_lineval, ckpt,folder):
    
    filename=folder
    folder = os.path.join('Figures/', folder)
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
    
    backbone_lineval = CNN().cuda()  # defining a raw backbone model
    checkpoint = torch.load(ckpt, map_location='cpu')
    backbone_lineval.load_state_dict(checkpoint)
    backbone_lineval.eval()
    acc_trains = list()
    
    X, y = [], []
    
    for data, target in test_loader_lineval:
        
        data = data.cuda()
        output = backbone_lineval(data)

        output=output.detach().cpu().numpy()
        output=normalize(output)
        X.append(output)
        
    
        # X.append(output.detach().cpu().numpy())
        y.append(target.detach().cpu().numpy())
        
    X = [item for sublist in X for item in sublist]
    y = [item for sublist in y for item in sublist]
    
    
    train_embeddings = folder+'embeddings'+ '.npy'
    train_labelsname = folder+'labels'+'.npy'
    np.save(train_embeddings,X, allow_pickle=True)
    np.save(train_labelsname,y, allow_pickle=True)
    
    graph_name1=folder+str(filename)+'_2D'+'.png'
    graph_name2=folder+str(filename)+'_3D'+'.png'
    
    ax,fig,graph_name=TSNEplot(X,y,graph_name1,graph_name2,str(filename),limits=2.6,perplexity=10)
    
    graph_name=folder+graph_name
    
    print(graph_name)
    
    def rotate(angle):
          ax.view_init(azim=angle)
    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(graph_name, writer=animation.PillowWriter(fps=20))
    
    return X, y,ax,fig,graph_name
 
def normalize(Features):
    
    df = pd.DataFrame(Features)  
    # df = df.apply(lambda x: (x - np.min(x))/(np.max(x)-np.min(x)), axis=1)
    df = df.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    df = df.to_numpy()
    return df
#%%
