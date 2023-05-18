# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:37:13 2022

@author: srpv
"""
import torch
import os
import math

from CNN_Saliency.Trainer.pytorchtools import EarlyStopping
from CNN_Saliency.Network.Model import CNN,LinearNetwork
from torch.optim.lr_scheduler import CosineAnnealingLR
from CNN_Saliency.Visualization.Training_plots import training_curves
from prettytable import PrettyTable

def supervised_train(train_loader_lineval, val_loader_lineval, test_loader_lineval, nb_class, opt,folder):
    
   
    num_steps_per_epoch = math.floor(len(train_loader_lineval.dataset) / opt.batch_size)
    print("Num_steps_per_epoch....",num_steps_per_epoch)   
    folder = os.path.join('Figures/', folder)
    
    try:
        os.makedirs(folder, exist_ok = True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")
    
    print(folder)
    folder=folder+'/'
   
    backbone_lineval = CNN(opt.feature_size).cuda()  # defining a raw backbone model
    linear_layer = LinearNetwork(opt.feature_size, len(nb_class)).cuda()
    optimizer = torch.optim.Adam([{'params': backbone_lineval.parameters()},
                  {'params': linear_layer.parameters()}], lr=opt.learning_rate)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * opt.epochs, eta_min=1e-4)
    CE = torch.nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(opt.patience_test, verbose=True,
                                   checkpoint_pth='{}backbone_best.tar'.format(folder))

    torch.save(backbone_lineval.state_dict(), '{}backbone_init.tar'.format(folder))
    torch.save(linear_layer.state_dict(), '{}linear_init.tar'.format(folder))
    
    best_acc = 0
    best_epoch = 0

    Training_loss=[]
    Training_accuracy=[]
    learn_rate=[]
    
    print('Supervised Train')
    for epoch in range(opt.epochs):
        backbone_lineval.train()
        linear_layer.train()
        
        acc_epoch=0
        acc_epoch_cls=0
        loss_epoch=0

        acc_trains = list()
        for i, (data, target) in enumerate(train_loader_lineval):
            
            scheduler.step()  
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()

            output = backbone_lineval(data)
            output = linear_layer(output)
            loss = CE(output, target)
            loss.backward()
            optimizer.step()
            # estimate the accuracy
            prediction = output.argmax(-1)
            correct = prediction.eq(target.view_as(prediction)).sum()
            accuracy = (100.0 * correct / len(target))
            acc_trains.append(accuracy.item())
            
            acc_epoch += accuracy.item()
            loss_epoch += loss.item()

        acc_epoch /= len(train_loader_lineval)
        loss_epoch /= len(train_loader_lineval)

        
        Training_loss.append(loss_epoch)
        Training_accuracy.append(acc_epoch)
        learn_rate.append(scheduler.get_last_lr()[0])

        print('[Train-{}][{}] loss: {:.5f}; \t Acc: {:.2f}%' \
              .format(epoch + 1, 'Supervised', loss.item(), sum(acc_trains) / len(acc_trains)))

        acc_vals = list()
        acc_tests = list()
        backbone_lineval.eval()
        linear_layer.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader_lineval):
                data = data.cuda()
                target = target.cuda()

                output = backbone_lineval(data).detach()
                output = linear_layer(output)
                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_vals.append(accuracy.item())

            val_acc = sum(acc_vals) / len(acc_vals)
            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = epoch
                for i, (data, target) in enumerate(test_loader_lineval):
                    data = data.cuda()
                    target = target.cuda()

                    output = backbone_lineval(data).detach()
                    output = linear_layer(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_tests.append(accuracy.item())

                test_acc = sum(acc_tests) / len(acc_tests)

        print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
            epoch+ 1, val_acc, test_acc, best_epoch))
        
        early_stopping(val_acc, backbone_lineval)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(backbone_lineval.state_dict(), '{}backbone_last.tar'.format(folder))
    torch.save(linear_layer.state_dict(), '{}linear_layer_last.tar'.format(folder))
    training_curves(Training_loss,Training_accuracy,learn_rate,folder)

    return backbone_lineval,linear_layer



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
        
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

