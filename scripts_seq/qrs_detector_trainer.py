#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:05:41 2019

@author: augusto
"""

import os
import pickle
import time
import matplotlib.pyplot as plt

import numpy as np

from progress.bar import ChargingBar

import qrs_detector

import torch

from torch.utils.data import DataLoader

class my_bar(ChargingBar):
    def __init__(self, epoch):
        super(my_bar, self).__init__(message = "Epoch {}".format(epoch),
                                     suffix = "Batch %(index)d/%(max)d - %(remaining_time)s - Loss: %(loss).4f - F1: %(F1).4f%%")
        self._loss = 0.0
        self._F1 = 0.0
    
    @property
    def remaining_time(self):
        h = self.eta // 3600
        m = (self.eta - h * 3600) // 60
        s = (self.eta - h * 3600 - m * 60) // 1
        
        ret = "{} hours {} min {} segs remaining".format(h, m, s)
        
        return ret
    
    @property
    def my_index(self):
        return self.index + 1
    
    @property
    def my_max(self):
        return self.max + 1
    
    @property
    def loss(self):
        return self._loss
    
    @loss.setter
    def loss(self, new_loss):
        self._loss = new_loss
    
    @property
    def F1(self):
        return self._F1 * 100
    
    @F1.setter
    def F1(self, new_F1):
        self._F1 = new_F1

if __name__ == "__main__":
    train_time_in_secs = 0 # Todo el dataset o cantidad de batches
    
    # Tamaño de cada batch
    batch_size = 512
    total_epochs = 5
    #batchs_per_epoch = math.ceil(500e3 / batch_size)
    batchs_per_epoch = 0 # Todo el dataset
    
    flag_make_fourier = False
    
    # Creacion de generadores tanto de entrenamiento como validacion
    train_gen = qrs_detector.dataset_loader_optim('/home/augusto/Desktop/GIBIO/processed_dbs/rata_segmentada_ia_v3',
                             '/home/augusto/Desktop/GIBIO/processed_dbs/rata_segmentada_ia_v3/train_set.txt')

    # Donde guardar el modelo
    model_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_rata_v3"
    model_filename = "qrs_det_model_rata_epoch_{}.pt"
    
    # Donde guardar historia del entrenamiento
    train_progress_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_rata_v3"
    train_progress_filename = "qrs_det_model_rata_training.bin"

    # Si num_workers no es cero, la RAM aumenta constantemente hasta que explota el SO
    train_dataloader = DataLoader(train_gen,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0)
    
    # Definicion del modelo
    model = qrs_detector.qrs_det_1_beta(train_gen.shape)
    
    loss_fn = torch.nn.BCELoss()
    
    learning_rate = 250e-6
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Entering training loop...")
    print("Total epochs: {}".format(total_epochs))
    
    if train_time_in_secs == 0 and batchs_per_epoch == 0:
        print("Whole training set mode.")
    elif batchs_per_epoch > 0:
        print("Batchs per epoch: {}".format(batchs_per_epoch))
    else:
        print("Training time per epoch: {}".format(train_time_in_secs))
    
    train_progress = []
    
    input("Press Enter to start.")
    
    for this_epoch in range(total_epochs):
        start_time = time.time()
        
        progress_bar = my_bar(this_epoch + 1)
        
        if batchs_per_epoch == 0:
            progress_bar.max = train_dataloader.__len__()
        else:
            progress_bar.max = batchs_per_epoch
        
        progress_bar.start()
        
        epoch_progress = []
        
        for this_batch, samples in enumerate(train_dataloader):
            if batchs_per_epoch > 0:
                if this_batch >= batchs_per_epoch:
                    break
            elif train_time_in_secs > 0:
                if time.time() - start_time > train_time_in_secs:
                    break
            
            batch_train_progress = {}
            
            batch_confusion_matrix = {}
            batch_confusion_matrix['TP'] = 0
            batch_confusion_matrix['TN'] = 0
            batch_confusion_matrix['FP'] = 0
            batch_confusion_matrix['FN'] = 0
            
            batch_loss = 0
            
            for this_sample, this_label in zip(samples[0], samples[1]):
                if flag_make_fourier:
                    this_sample_np = this_sample.view(1,24).numpy()
                    this_sample_fft = torch.from_numpy(np.float32(np.abs(np.fft.fft(this_sample_np)))).view(1,1,24)
                    y_pred = model(this_sample_fft)
                else:
                    y_pred = model(this_sample)
                
                this_loss = loss_fn(y_pred, this_label)
                batch_loss += this_loss.item()
                optimizer.zero_grad()
                this_loss.backward()
                optimizer.step()
            
                if y_pred.item() >= 0.5:
                    # La IA interpreto que hay latido
                    if this_label.item() == 1:
                        # Habia latido, TP
                        batch_confusion_matrix['TP'] += 1
                    else:
                        # No habia latido, FN
                        batch_confusion_matrix['FP'] += 1
                else:
                    # La IA interpreto que no hay latido
                    if this_label.item() == 1:
                        # Habia latido, FP
                        batch_confusion_matrix['FN'] += 1
                    else:
                        # No habia latido, TN
                        batch_confusion_matrix['TN'] += 1
            
            batch_train_progress['confusion_matrix'] = batch_confusion_matrix
            batch_train_progress['specifiers'] = qrs_detector.make_specifiers(batch_confusion_matrix)
            batch_train_progress['loss'] = batch_loss / batch_size
            
            progress_bar.F1 = batch_train_progress['specifiers']['F1']
            progress_bar.loss = batch_loss / batch_size
            
            progress_bar.next()
            
            """
            if batch_train_progress['specifiers']['MCC'] <= -0.2:
                print(Back.WHITE + Fore.RED + "Batch idx: {}/{} - Batch loss (average): {:.5f}".format(this_batch,
                                                                     batchs_per_epoch if batchs_per_epoch > 0 else train_dataloader.__len__(),
                                                                     batch_loss / batch_size))
                print(Back.WHITE + Fore.RED + "MCC: {:.4f} --- ACC: {:.4f} --- F1: {:.5f}".format(batch_train_progress['specifiers']['MCC'],
                                                                     batch_train_progress['specifiers']['ACC'],
                                                                     batch_train_progress['specifiers']['F1']))
            elif batch_train_progress['specifiers']['MCC'] <= 0.6:
                print(Back.WHITE + Fore.BLUE + "Batch idx: {}/{} - Batch loss (average): {:.5f}".format(this_batch,
                                                                     batchs_per_epoch if batchs_per_epoch > 0 else train_dataloader.__len__(),
                                                                     batch_loss / batch_size))
                print(Back.WHITE + Fore.BLUE + "MCC: {:.4f} --- ACC: {:.4f} --- F1: {:.5f}".format(batch_train_progress['specifiers']['MCC'],
                                                                     batch_train_progress['specifiers']['ACC'],
                                                                     batch_train_progress['specifiers']['F1']))
            else:
                print(Back.WHITE + Fore.GREEN + "Batch idx: {}/{} - Batch loss (average): {:.5f}".format(this_batch,
                                                                     batchs_per_epoch if batchs_per_epoch > 0 else train_dataloader.__len__(),
                                                                     batch_loss / batch_size))
                print(Back.WHITE + Fore.GREEN + "MCC: {:.4f} --- ACC: {:.4f} --- F1: {:.5f}".format(batch_train_progress['specifiers']['MCC'],
                                                                     batch_train_progress['specifiers']['ACC'],
                                                                     batch_train_progress['specifiers']['F1']))
            
            print(Back.WHITE + Fore.BLACK + "Elapsed time: {:.0f} secs".format(time.time() - start_time))
            """
            
            epoch_progress.append(batch_train_progress)
        
        train_progress.append(epoch_progress)
        torch.save(model.state_dict(), os.path.join(model_path, model_filename.format(this_epoch + 1)))
        progress_bar.finish()
    
    pickle.dump(train_progress, open(os.path.join(train_progress_path, train_progress_filename), 'wb'))
    input("Press enter to finish...")