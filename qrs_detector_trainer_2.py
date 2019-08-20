#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:16:37 2019

@author: augusto
"""

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
    train_gen = qrs_detector.dataset_loader('/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_beta',
                             '/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_beta/train_set.txt')

    # Donde guardar el modelo
    model_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia"
    model_filename = "qrs_det_model_2.pt"
    
    # Donde guardar historia del entrenamiento
    train_progress_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia"
    train_progress_filename = "qrs_det_model_2_training.bin"
    
    progress_file = open(os.path.join(train_progress_path, train_progress_filename), 'wb')

    train_dataloader = DataLoader(train_gen,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 3)
    
    # Definicion del modelo
    model = qrs_detector.qrs_det_2()
    
    loss_fn = torch.nn.MSELoss(reduction = 'sum')
    
    learning_rate = 1e-3
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Entering training loop...")
    
    batch_average_diff = []
    
    for this_epoch in range(total_epochs):
        print(Fore.BLACK + "Epoch: {}/{}".format(this_epoch, total_epochs))
        
        average_diff_tracker = []
        
        for this_batch, samples in enumerate(train_dataloader):
            if this_batch >= batchs_per_epoch:
                break
            
            y_pred = model(samples[0].view(batch_size, 1, -1))
            loss = loss_fn(y_pred, samples[1].view(batch_size, 1, -1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            diff_tracker = [abs(y_pred[aux].item() - samples[1][aux]) for aux in range(len(samples[0]))]
            
            average_diff = sum(diff_tracker) / len(diff_tracker)
            max_diff = max(diff_tracker)
            
            average_diff_tracker.append(average_diff)
            
            if max_diff >= 0.5:
                print(Fore.RED + "Batch idx: {}/{} - Average diff: {}".format(this_batch, batchs_per_epoch, average_diff))
                print("Max diff: {}".format(max_diff))
            elif max_diff >= 0.2:
                print(Fore.BLUE + "Batch idx: {}/{} - Average diff: {}".format(this_batch, batchs_per_epoch, average_diff))
                print("Max diff: {}".format(max_diff))
            else:
                print(Fore.GREEN + "Batch idx: {}/{} - Average diff: {}".format(this_batch, batchs_per_epoch, average_diff))
                print("Max diff: {}".format(max_diff))
        
        batch_average_diff.append(average_diff_tracker)
    
    torch.save(model.state_dict(), os.path.join(model_path, model_filename))
    pickle.dump(batch_average_diff, progress_file)