#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:22:38 2019

@author: augusto
"""

import os
import pickle

import qrs_detector

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Archivo de entrenamiento
    train_progress_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia"
    train_progress_filename = "qrs_det_model_3_training_nofilter.bin"
    
    train_progress = pickle.load(open(os.path.join(train_progress_path, train_progress_filename),'rb'))
    
    epochs = np.arange(start = 1, stop = 11)
    specifiers = {}
    specifiers['MCC'] = []
    specifiers['ACC'] = []
    specifiers['F1'] = []
    
    for this_train in train_progress:
        this_epoch_cm = {}
        this_epoch_cm['TP'] = 0
        this_epoch_cm['TN'] = 0
        this_epoch_cm['FP'] = 0
        this_epoch_cm['FN'] = 0
        
        for this_batch in this_train:
            this_epoch_cm['TP'] += this_batch['confusion_matrix']['TP']
            this_epoch_cm['TN'] += this_batch['confusion_matrix']['TN']
            this_epoch_cm['FP'] += this_batch['confusion_matrix']['FP']
            this_epoch_cm['FN'] += this_batch['confusion_matrix']['FN']
        
        this_epoch_specifiers = qrs_detector.make_specifiers(this_epoch_cm)
        
        specifiers['MCC'].append(this_epoch_specifiers['MCC'])
        specifiers['ACC'].append(this_epoch_specifiers['ACC'])
        specifiers['F1'].append(this_epoch_specifiers['F1'])
    
    plt.plot(epochs, specifiers['F1'], epochs, specifiers['ACC'], epochs, specifiers['MCC'])
    