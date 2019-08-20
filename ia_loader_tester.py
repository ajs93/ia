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
    
    x = torch.randn(1, 1, 24)
    y = torch.ones(batch_size, 1, 1, 1)
    
    y_pred = model(x)