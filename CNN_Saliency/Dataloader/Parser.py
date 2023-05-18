# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:09:53 2023

@author: srpv
"""

import argparse
def parse_option():
    
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    

    parser.add_argument('--feature_size', type=int, default=64,
                        help='feature_size')
    
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    
    parser.add_argument('--patience', type=int, default=400,
                        help='training patience')
  
   

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    
    
    # model dataset
    
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Data path for checkpoint.')
    
    # Testing
    parser.add_argument('--learning_rate_test', type=float, default=0.01,
                        help='learning_rate_test')
    
    parser.add_argument('--patience_test', type=int, default=100,
                        help='number of training patience')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    opt = parser.parse_args()
    return opt

