#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:05:41 2019

@author: augusto
"""

import os
import pickle

from colorama import Fore

import qrs_detector

import torch

from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Tamaño de cada batch
    batch_size = 256
    total_epochs = 10
    batchs_per_epoch = 100
    
    # Creacion de generadores tanto de entrenamiento como validacion
    train_gen = qrs_detector.dataset_loader('/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_nofilter',
                             '/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_nofilter/train_set.txt')

    # Donde guardar el modelo
    model_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia"
    model_filename = "qrs_det_model_3_nofilter.pt"
    
    # Donde guardar historia del entrenamiento
    train_progress_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia"
    train_progress_filename = "qrs_det_model_3_training_nofilter.bin"
    
    progress_file = open(os.path.join(train_progress_path, train_progress_filename), 'wb')

    train_dataloader = DataLoader(train_gen,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 3)
    
    # Definicion del modelo
    model = qrs_detector.qrs_det_3()
    
    loss_fn = torch.nn.MSELoss(reduction = 'sum')
    
    learning_rate = 1e-3
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Entering training loop...")
    
    train_progress = []
    
    for this_epoch in range(total_epochs):
        print(Fore.BLACK + "Epoch: {}/{}".format(this_epoch, total_epochs))
        
        epoch_progress = []
        
        for this_batch, samples in enumerate(train_dataloader):
            if this_batch >= batchs_per_epoch:
                break
            
            batch_train_progress = {}
            
            batch_confusion_matrix = {}
            batch_confusion_matrix['TP'] = 0
            batch_confusion_matrix['TN'] = 0
            batch_confusion_matrix['FP'] = 0
            batch_confusion_matrix['FN'] = 0
            
            for this_idx in range(len(samples[0])):
                y_pred = model(samples[0][this_idx])
                this_loss = loss_fn(y_pred, samples[1][this_idx])
                optimizer.zero_grad()
                this_loss.backward()
                optimizer.step()
            
                if y_pred.item() >= 0.5:
                    # La IA interpreto que hay latido
                    if samples[1][this_idx].item() == 1:
                        # Habia latido, TP
                        batch_confusion_matrix['TP'] += 1
                    else:
                        # No habia latido, FN
                        batch_confusion_matrix['FN'] += 1
                else:
                    # La IA interpreto que no hay latido
                    if samples[1][this_idx].item() == 1:
                        # Habia latido, FP
                        batch_confusion_matrix['FP'] += 1
                    else:
                        # No habia latido, TN
                        batch_confusion_matrix['TN'] += 1
            
            batch_train_progress['confusion_matrix'] = batch_confusion_matrix
            batch_train_progress['specifiers'] = qrs_detector.make_specifiers(batch_confusion_matrix)
            
            if batch_train_progress['specifiers']['MCC'] <= -0.3:
                print(Fore.RED + "Batch idx: {}/{} - MCC: {}".format(this_batch, batchs_per_epoch, batch_train_progress['specifiers']['MCC']))
            elif batch_train_progress['specifiers']['MCC'] <= 0.3:
                print(Fore.BLUE + "Batch idx: {}/{} - MCC: {}".format(this_batch, batchs_per_epoch, batch_train_progress['specifiers']['MCC']))
            else:
                print(Fore.GREEN + "Batch idx: {}/{} - MCC: {}".format(this_batch, batchs_per_epoch, batch_train_progress['specifiers']['MCC']))
            
            epoch_progress.append(batch_train_progress)
        
        train_progress.append(epoch_progress)
    
    torch.save(model.state_dict(), os.path.join(model_path, model_filename))
    pickle.dump(train_progress, progress_file)