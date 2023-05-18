# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:00:09 2023

@author: srpv
"""


import torch
import numpy as np
from matplotlib import pyplot as plt
from EMD import ComputeEMD
from scipy.signal import chirp
import IPython
import pandas as pd
from utils import *

'''
Download the dataset from following link
-- https://polybox.ethz.ch/index.php/s/9xLsB1kJORolfTc
'''

sample_rate = 400000
windowsize= 5000
N=windowsize
t0=0
dt=1/sample_rate
time = np.arange(0, N) * dt + t0
duration = windowsize/sample_rate
t = torch.arange(sample_rate*duration) / sample_rate
t=t.detach().cpu().numpy().squeeze()

#%%


sensor=['D1','D2','D3','D4']
regimes=['LoF pores' ,'Conduction mode','Keyhole pores']
path=r'C:\Users\srpv\Desktop\LPBF Sensor selection Decomposition\Data' #Data folder path


for k in sensor:
    for z in regimes:
    

        dataset = k
        class_1 = 'LoF pores' 
        class_2 = 'Conduction mode'
        class_3 = 'Keyhole pores'
        
        print(k)
        print(z)
        
        if z=='LoF pores':
            color='blue'
            
        elif z=='Conduction mode':
            color='green'
            
        else:
            color='red'
            
        
        classname=z

        x = Normal_Regime(path,dataset, classname,windowsize) #Select the normal regimes
        
        print(x.shape)
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        fig = plt.figure(figsize = (10, 3))
        plt.plot(t,x.ravel(),color) 
        plt.title(str(dataset)+'-'+str(classname))
        plt.xlabel("Time (sec)",fontsize=15)
        plt.ylabel("Amplitude (V)",fontsize=15)
        plotname=str(dataset)+'_'+str(classname)+'_rawsignal.png'
        plt.savefig(plotname,dpi=200)
        plt.show()



        imfs = ComputeEMD(x, sample_rate, num_imf=7)
        imfs=imfs.detach().cpu().numpy().squeeze()
        
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.figure(figsize=(5,12))
        
        for i in range(len(imfs)):
            plt.subplot(len(imfs),1,i+1)
            
            # min=np.min(x.ravel())
            # max=np.max(x.ravel())
            
            plt.plot(t,x.ravel(),'black',color='0.6')  
            plt.plot(t,imfs[i],color)
        
            # plt.ylim([min*1.1,max*1.1])
            plt.ylabel('IMF '+np.str(i+1),fontsize=15)
            
            if i != len(imfs)-1:
                
                plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) 
            
            if i == len(imfs)-1:
                plt.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
                plt.xlabel('Time (s)',fontsize=15)
                
            if i == 0:
                plt.title('Raw signal vs IMFs',fontsize=15)
                # plt.legend(['Rawsignal','IMF(s)'],loc='lower right',frameon=False, bbox_to_anchor=(1.0, 1.1))
                
        plt.tight_layout()
        plotname=str(dataset)+'_'+str(classname)+'_IMF.png'        
        plt.savefig(plotname,dpi=400) 
        plt.show()   
        plt.close()

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        fig, axs = plt.subplots(nrows=len(imfs), ncols=1, sharex=True,
                                            figsize=(6, 12),dpi=400)
        
        for i in range(len(imfs)):
            ax = axs.flat[i]
            Frequencyplot(imfs,sample_rate,i,ax)
            if i == 0:
                skyblue = mpatches.Patch(color='#657e39', label='0-30 kHz')
                # ax.legend(handles=[skyblue])
                red = mpatches.Patch(color='#aa332d', label='30-60 kHz')
                # ax.legend(handles=[red])
                yellow = mpatches.Patch(color='#f0a334', label='60-90 kHz')
                # ax.legend(handles=[yellow])
                green = mpatches.Patch(color='#0080ff', label='90-120 kHz')
                # ax.legend(handles=[green])
                cyan = mpatches.Patch(color='#b05aac', label='120-150 kHz')
                
                ax.set_title('IMF vs Frequency',fontsize=15)
                
            if i != len(imfs)-1:    
                # ax.set_xticks([])
                plt.setp( ax.get_xticklabels(), visible=False)
                # plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
                
            if i == len(imfs)-1:
                # plt.setp( ax.get_xticklabels(), visible=True)
                plt.xlabel('Frequency(Hz)',fontsize=15)
                # ax.legend(handles=[skyblue,red,yellow,green,cyan], bbox_to_anchor=(1, 0.35))
        plt.tight_layout() 
        plotname=str(dataset)+'_'+str(classname)+ '_IMF_FFT.png'      
        plt.savefig(plotname,dpi=400) 
        plt.show()   