#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:00:12 2019

@author: augusto
"""

import os
import pickle
import torch
import math

from torch.utils.data import Dataset

class dim_error(Exception):
    def __str__(self):
        return 'Different dimensions in dataset.'

class dataset_loader_rnn(Dataset):
    def __init__(self, database_source_path, target_file_path):
        # Abro archivo de entrenamiento pasado y obtengo lista de archivos
        target_file = open(target_file_path, 'r')
        
        self.file_names = target_file.read()
        
        self.file_names = self.file_names.split('\n')
        
        # Eliminacion de elementos nulos
        self.file_names = list(filter(None, self.file_names))
        
        if not database_source_path[-1] == os.sep:
            database_source_path += os.sep
        
        # Acomodo rutas en base a directorio base
        self.file_names = [database_source_path + s for s in self.file_names]
        
        target_file.close()
        
        # En la inicializacion, obtengo el total de datos de entrenamiento disponible
        self.file_limits = []
        self.total_slices = 0
        self.dim = None
        self.current_file_idx = -1
        self.current_data = None
        
        for this_filename in self.file_names:
            this_data = pickle.load(open(this_filename, 'rb'))
            
            # Del primero saco el shape, deberia ser igual en todos
            if self.dim is None:
                self.dim = len(this_data['slices'][0])
            elif self.dim != len(this_data['slices'][0]):
                raise dim_error()
            
            self.total_slices += len(this_data['slices'])
            self.file_limits.append(self.total_slices)
        
    def __len__(self):
        return self.total_slices
    
    def __getitem__(self, idx):
        # Obtengo indice interno
        file_idx = 0
        
        while idx >= self.file_limits[file_idx]:
            file_idx += 1
        
        # Como el entrenador va a ir en orden, mantengo un archivo abierto
        if file_idx != self.current_file_idx:
            # Si es un archivo distinto, abro el archivo nuevo
            self.current_data = pickle.load(open(self.file_names[int(file_idx)], 'rb'))        
        
        # Con el indice real, devuelvo el dato pedido
        if file_idx > 0:
            real_idx = idx - self.file_limits[file_idx - 1]
        else:
            real_idx = idx
        
        x = torch.Tensor(self.current_data['slices'][real_idx])
        x = x.view(1, 1, 24)
        
        if self.current_data['labels'][real_idx] == 0:
            y = torch.zeros(1, 1, 1)
        else:
            y = torch.ones(1, 1, 1)
        
        return x, y

class qrs_det_rnn(torch.nn.Module):
    def __init__(self, input_dim):
        super(qrs_det_rnn, self).__init__()
        
        self.input_dim = input_dim
        
        self.rnn = torch.nn.RNN(input_size = input_dim,
                                hidden_size = input_dim,
                                num_layers = 4,
                                nonlinearity = 'relu',
                                dropout = 0.1)
        
        self.lin = torch.nn.Linear(in_features = input_dim,
                                   out_features = 1)
        
        self.sig = torch.nn.Sigmoid()
        
    def forward(self, x, h):
        if h is None:
            x, hidden = self.rnn(x, h)
        else:
            x, hidden = self.rnn(x, h.detach())
        
        x = self.lin(x)
        x = self.sig(x)
        
        return x, hidden
    
def make_specifiers(cm):
    """
    Funcion para generar indicadores de un algoritmo de deteccion
    
    Parameters
    ----------
    cm : dictionary
    Confusion Matrix result of the algorithm.
        TP : True Positives
        TN : True Negatives
        FP : False Positives
        FN : False Negatives
    
    Returns
    ----------
    specifiers : dictionary
        TPR : Sensitivity (TP/(TP + FN))
        TNR : Specificity (TN/(TN + FP))
        PPV : Precision (TP/(TP + FP))
        NPV : Negative predictive value (TN/(TN + FN))
        ACC : Accuracy ((TP + TN)/(TP + TN + FP + FN))
        F1 : F1 Score ((2 * TP)/(2 * TP + FP + FN))
        MCC : Matthews correlation coefficient ((TP * TN - FP * FN)/sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN)))
    """
    specifiers = {}
    
    specifiers['TPR'] = cm['TP'] / (cm['TP'] + cm['FN']) if cm['TP'] + cm['FN'] else 0
    specifiers['TNR'] = cm['TN'] / (cm['TN'] + cm['FP']) if cm['TN'] + cm['FP'] else 0
    specifiers['PPV'] = cm['TP'] / (cm['TP'] + cm['FP']) if cm['TP'] + cm['FP'] else 0
    specifiers['NPV'] = cm['TN'] / (cm['TN'] + cm['FN']) if cm['TN'] + cm['FN'] else 0
    specifiers['ACC'] = (cm['TP'] + cm['TN']) / (cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']) if cm['TP'] + cm['TN'] + cm['FP'] + cm['FN'] else 0
    specifiers['F1'] = (2 * cm['TP']) / (2 * cm['TP'] + cm['FP'] + cm['FN']) if 2 * cm['TP'] + cm['FP'] + cm['FN'] else 0
    specifiers['MCC'] = (cm['TP'] * cm['TN'] - cm['FP'] * cm['FN']) / math.sqrt((cm['TP'] + cm['FP']) * (cm['TP'] + cm['FN']) * (cm['TN'] + cm['FP']) * (cm['TN'] + cm['FN'])) if math.sqrt((cm['TP'] + cm['FP']) * (cm['TP'] + cm['FN']) * (cm['TN'] + cm['FP']) * (cm['TN'] + cm['FN'])) else 0
    
    return specifiers