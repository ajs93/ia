#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:05:41 2019

@author: augusto
"""

import os
import pickle
import time

from colorama import Fore

import qrs_detector

import torch

from torch.utils.data import DataLoader

if __name__ == "__main__":
    start_time = time.time()
    train_time_in_secs = 6 * 60 * 60 # 6 horas
    
    # Tamaño de cada batch
    batch_size = 4096
    total_epochs = 1
    batchs_per_epoch = 0
    
    # Creacion de generador de entrenamiento
    train_gen = qrs_detector.dataset_loader_rnn('/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_rnn',
                             '/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_rnn/train_set.txt')

    # Donde guardar el modelo
    model_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_rnn"
    model_filename = "qrs_det_model_rnn.pt"
    
    # Donde guardar historia del entrenamiento
    train_progress_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_rnn"
    train_progress_filename = "qrs_det_model_rnn_training.bin"

    train_dataloader = DataLoader(train_gen,
                                  batch_size = batch_size,
                                  shuffle = False,
                                  num_workers = 3)
    
    # Definicion del modelo
    model = qrs_detector.qrs_det_rnn(train_gen.dim)
    
    loss_fn = torch.nn.MSELoss(reduction = 'sum')
    
    learning_rate = 100e-6
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Entering training loop...")
    
    train_progress = []
    hidden_state = None
    
    for this_epoch in range(total_epochs):
        print(Fore.BLACK + "Epoch: {}/{}".format(this_epoch, total_epochs))
        
        epoch_progress = []
        
        for this_batch, samples in enumerate(train_dataloader):
            if batchs_per_epoch > 0:
                if this_batch >= batchs_per_epoch:
                    break
            else:
                if time.time() - start_time > train_time_in_secs:
                    break
            
            batch_train_progress = {}
            
            batch_confusion_matrix = {}
            batch_confusion_matrix['TP'] = 0
            batch_confusion_matrix['TN'] = 0
            batch_confusion_matrix['FP'] = 0
            batch_confusion_matrix['FN'] = 0
            
            for this_idx in range(len(samples[0])):
                y_pred, hidden_state = model(samples[0][this_idx], hidden_state)
                this_loss = loss_fn(y_pred, samples[1][this_idx])
                optimizer.zero_grad()
                this_loss.backward()
                optimizer.step()
                
                #print("loss: {}".format(this_loss.item()))
                
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
            
            if batch_train_progress['specifiers']['F1'] <= 0.65:
                print(Fore.RED + "Batch idx: {}/{} - F1: {}".format(this_batch,
                                                                     batchs_per_epoch if batchs_per_epoch > 0 else train_dataloader.__len__(),
                                                                     batch_train_progress['specifiers']['F1']))
            elif batch_train_progress['specifiers']['F1'] <= 0.9:
                print(Fore.BLUE + "Batch idx: {}/{} - F1: {}".format(this_batch,
                                                                      batchs_per_epoch if batchs_per_epoch > 0 else train_dataloader.__len__(),
                                                                      batch_train_progress['specifiers']['F1']))
            else:
                print(Fore.GREEN + "Batch idx: {}/{} - F1: {}".format(this_batch,
                                                                       batchs_per_epoch if batchs_per_epoch > 0 else train_dataloader.__len__(),
                                                                       batch_train_progress['specifiers']['F1']))
            
            print(Fore.BLACK + "Elapsed time: {} secs".format(time.time() - start_time))
            
            epoch_progress.append(batch_train_progress)
        
        train_progress.append(epoch_progress)
    
    torch.save(model.state_dict(), os.path.join(model_path, model_filename))
    pickle.dump(train_progress, open(os.path.join(train_progress_path, train_progress_filename), 'wb'))