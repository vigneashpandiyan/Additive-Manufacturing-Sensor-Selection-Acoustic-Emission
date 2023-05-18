# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:56:21 2023

@author: srpv
"""

from torch.utils.data import Dataset
import torch

class data_tensor(Dataset):
    
    def __init__(self,sequences):
        self.sequences = sequences
    
    def __len__(self):
        
        return len(self.sequences)
    
    def __getitem__(self,idx):
        
        sequence_0,label =  self.sequences [idx]
        
        sequence_1=torch.Tensor(sequence_0[0])
        sequence_2=torch.Tensor(sequence_0[1])
        sequence_3=torch.Tensor(sequence_0[2])
        sequence_4=torch.Tensor(sequence_0[3])
        sequence_5=torch.Tensor(sequence_0[4])
        sequence_6=torch.Tensor(sequence_0[5])
        sequence_7=torch.Tensor(sequence_0[6])
        
        sequence1 = sequence_1.view(1, -1)
        sequence2 = sequence_2.view(1, -1)
        sequence3 = sequence_3.view(1, -1)
        sequence4 = sequence_4.view(1, -1)
        sequence5 = sequence_5.view(1, -1)
        sequence6 = sequence_6.view(1, -1)
        sequence7 = sequence_7.view(1, -1)
        
        
        sequence=torch.cat((sequence1, sequence2,sequence3, sequence4,sequence5,sequence6,sequence7), 0)
        # print("sequence",sequence.shape)
        label=torch.tensor(label).long()
        # sequence,label
        return sequence,label

