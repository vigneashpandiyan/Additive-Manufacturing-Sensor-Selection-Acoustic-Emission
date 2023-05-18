# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:09:53 2023

@author: srpv
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrintLayer(torch.nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        # print(x.shape)
        return x

class CNN(nn.Module): 
    def __init__(self,feature_size=64):
        super(CNN, self).__init__()
        self.feature_size = feature_size
        self.dropout=0.05
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=7, out_channels=4, kernel_size=16, bias=False),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            PrintLayer())
            
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=16, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            PrintLayer())
            
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            PrintLayer())
            
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            PrintLayer())
            
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(self.dropout),
            # nn.MaxPool1d(3),
            torch.nn.AdaptiveAvgPool1d(1),
            PrintLayer())
            
        
               
    def forward(self, x):
        
        
        x = self.conv1 (x)
        x = self.conv2 (x)
        x = self.conv3 (x)
        x = self.conv4 (x)
        x = self.conv5 (x)
    
        x= x.view(x.size(0), -1)
        
        return x


class LinearNetwork(nn.Module): 
    def __init__(self,nb_class,feature):
        super(LinearNetwork, self).__init__()
        self.feature = feature
        self.nb_class=nb_class
        
        
        self.fc1=nn.Linear(64, 32)
        self.fc2=nn.Linear(32, 3)
        
        self.conv1 = nn.Sequential(
            nn.ReLU(),
           )
        
    def forward(self, x):
    
        x=self.fc1(x)
        x = self.conv1(x)
        x=self.fc2(x)
        
        return x