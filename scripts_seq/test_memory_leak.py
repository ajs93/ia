#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:05:41 2019

@author: augusto
"""

import os
import pickle
import time

from colorama import Fore, Back

import qrs_detector

import torch

from torch.utils.data import DataLoader

if __name__ == "__main__":
    start_time = time.time()
    train_time_in_secs = 0 # todo el dataset
    
    # Tamaño de cada batch
    batch_size = 4096
    total_epochs = 10
    batchs_per_epoch = 5
    
    # Creacion de generadores tanto de entrenamiento como validacion
    train_gen = qrs_detector.dataset_loader_optim('/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_agosto',
                             '/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_agosto/train_set.txt')

    # Donde guardar el modelo
    model_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_1_aux"
    model_filename = "qrs_det_model_1.pt"
    
    # Donde guardar historia del entrenamiento
    train_progress_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_1_aux"
    train_progress_filename = "qrs_det_model_1_training.bin"

    train_dataloader = DataLoader(train_gen,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 2)
    
    # Definicion del modelo
    model = qrs_detector.qrs_det_1_beta(train_gen.shape)
    
    print("Entering test loop...")
    
    train_progress = []
    
    for this_idx, this_batch in enumerate(train_dataloader):
        for this_sample, this_label in zip(this_batch[0], this_batch[1]):
            y = model(this_sample)
            y.detach()
        
        print("Computed {}/{} batches".format(this_idx, train_dataloader.__len__()))