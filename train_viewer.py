#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:22:38 2019

@author: augusto
"""

import os
import pickle

from matplotlib.widgets import MultiCursor
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Archivo de entrenamiento
    train_progress_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_1_beta_large"
    train_progress_filename = "qrs_det_model_1_training.bin"
    
    train_progress = pickle.load(open(os.path.join(train_progress_path, train_progress_filename),'rb'))
    
    this_plot, (ax1, ax2) = plt.subplots(2, sharex = True)
    
    this_plot.suptitle("Model: " + train_progress_filename[:-4])
    
    batch_number = []
    
    for epoch_idx, this_epoch in enumerate(train_progress):
        loss_progress = []
        F1_progress = []
        
        for batch_idx, this_batch in enumerate(this_epoch):
            F1_progress.append(this_batch['specifiers']['F1'])
            loss_progress.append(this_batch['loss'])
            batch_number.append(batch_idx + epoch_idx * len(this_epoch))
        
        ax1.plot(batch_number[epoch_idx * len(this_epoch):epoch_idx * len(this_epoch) + len(this_epoch)], F1_progress, lw = 1)
        ax2.plot(batch_number[epoch_idx * len(this_epoch):epoch_idx * len(this_epoch) + len(this_epoch)], loss_progress, lw = 1)
        
    ax1.grid(True)
    ax1.title.set_text("F1")
    ax1.set_ylim([0,1])
    
    ax2.grid(True)
    ax2.title.set_text("Loss")
    ax2.set_ylim([0,1])
    
    multi = MultiCursor(this_plot.canvas, (ax1, ax2), lw = 1)
    
    plt.show()