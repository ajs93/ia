#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:50:20 2019

@author: augusto
"""

import os
import pickle
import math

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

class dataset_loader(Dataset):
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
        self.slice_limits = []
        self.total_slices = 0
        self.shape = ()
        
        for this_filename in self.file_names:
            this_file = open(this_filename, 'rb')
            
            this_data = pickle.load(this_file)
            
            # Del primero saco el shape, deberia ser igual en todos
            if len(self.shape) == 0:
                self.shape = (this_data['beat_slices'][0][1] - this_data['beat_slices'][0][0], this_data['data'].shape[1])
            
            aux_slice_counter = len(this_data['beat_slices'])
            aux_slice_counter += len(this_data['no_beat_slices'])
            
            self.total_slices += aux_slice_counter
            self.slice_limits.append(self.total_slices)
            
            this_file.close()
        
    def __len__(self):
        return self.total_slices
    
    def __getitem__(self, idx):
        # Obtengo indice interno
        file_idx = 0
        
        while idx >= self.slice_limits[file_idx]:
            file_idx += 1
        
        # Obtengo datos del idx pedido
        this_data = pickle.load(open(self.file_names[int(file_idx)], 'rb'))
        
        # Con el indice real, devuelvo el dato pedido
        if file_idx > 0:
            real_idx = idx - self.slice_limits[file_idx - 1]
        else:
            real_idx = idx
        
        if real_idx <= len(this_data['beat_slices']) - 1:
            # Si es latido, la etiqueta es 1
            this_start = this_data['beat_slices'][real_idx][0]
            this_end = this_data['beat_slices'][real_idx][1]
            y = torch.ones(1, 1, 1)
        else:
            # Si es no latido, la etiqueta es 0
            real_idx -= len(this_data['beat_slices'])
            
            this_start = this_data['no_beat_slices'][real_idx][0]
            this_end = this_data['no_beat_slices'][real_idx][1]
            y = torch.zeros(1, 1, 1)
            
        x = torch.Tensor(this_data['data'][this_start:this_end])
        x = x.view(1, 1, 24)
        
        return x, y

class qrs_det_1(torch.nn.Module):
    def __init__(self):
        super(qrs_det_1, self).__init__()
        
        input_dim = 24
        
        conv1_kernel_size = 6
        conv2_kernel_size = 3
        
        lin0_input_size = input_dim
        lin0_output_size = round(input_dim * 2.5)
        
        lin1_input_size = lin0_output_size - conv1_kernel_size - conv2_kernel_size + 2
        lin1_output_size = input_dim
        
        lin2_input_size = lin1_output_size
        lin2_output_size = 12
        
        lin3_input_size = lin2_output_size
        lin3_output_size = 1
        
        self.lin0 = torch.nn.Linear(lin0_input_size, lin0_output_size)
        self.lin1 = torch.nn.Linear(lin1_input_size, lin1_output_size)
        self.lin2 = torch.nn.Linear(lin2_input_size, lin2_output_size)
        self.lin3 = torch.nn.Linear(lin3_input_size, lin3_output_size)
        
        self.norm = torch.nn.BatchNorm1d(1)
        
        self.conv1 = torch.nn.Conv1d(1, 1, conv1_kernel_size)
        self.conv2 = torch.nn.Conv1d(1, 1, conv2_kernel_size)
    
    def forward(self, x):
        x = self.norm(F.relu(self.lin0(x)))
        x = F.relu(F.dropout(self.conv1(x), p = 0.1))
        x = F.relu(F.dropout(self.conv2(x), p = 0.15))
        x = F.dropout(F.relu(self.lin1(x)), p = 0.2)
        x = F.dropout(F.relu(self.lin2(x)), p = 0.5)
        x = F.sigmoid(self.lin3(x))
        
        return x

class qrs_det_2(torch.nn.Module):
    def __init__(self):
        super(qrs_det_2, self).__init__()
        
        input_dim = 24
        
        self.conv_amount = 10
        conv_kernel_size = 6
        conv_padding = (int)(conv_kernel_size / 2)
        
        if input_dim % 2 == 0:
            lin1_input_size = input_dim + self.conv_amount
        else:
            lin1_input_size = input_dim
        
        lin1_output_size = input_dim
        
        lin2_input_size = lin1_output_size
        lin2_output_size = 12
        
        lin3_input_size = lin2_output_size
        lin3_output_size = 1
        
        self.conv0 = torch.nn.Conv1d(1, 1, conv_kernel_size, padding = conv_padding)
        self.conv1 = torch.nn.Conv1d(1, 1, conv_kernel_size, padding = conv_padding)
        self.conv2 = torch.nn.Conv1d(1, 1, conv_kernel_size, padding = conv_padding)
        self.conv3 = torch.nn.Conv1d(1, 1, conv_kernel_size, padding = conv_padding)
        self.conv4 = torch.nn.Conv1d(1, 1, conv_kernel_size, padding = conv_padding)
        self.conv5 = torch.nn.Conv1d(1, 1, conv_kernel_size, padding = conv_padding)
        self.conv6 = torch.nn.Conv1d(1, 1, conv_kernel_size, padding = conv_padding)
        self.conv7 = torch.nn.Conv1d(1, 1, conv_kernel_size, padding = conv_padding)
        self.conv8 = torch.nn.Conv1d(1, 1, conv_kernel_size, padding = conv_padding)
        self.conv9 = torch.nn.Conv1d(1, 1, conv_kernel_size, padding = conv_padding)
        
        self.lin1 = torch.nn.Linear(lin1_input_size, lin1_output_size)
        self.lin2 = torch.nn.Linear(lin2_input_size, lin2_output_size)
        self.lin3 = torch.nn.Linear(lin3_input_size, lin3_output_size)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
    
        x = F.dropout(F.relu(self.lin1(x)), p = 0.5)
        x = F.dropout(F.relu(self.lin2(x)), p = 0.2)
        x = F.sigmoid(self.lin3(x))
        
        return x
    
class qrs_det_3(torch.nn.Module):
    def __init__(self):
        super(qrs_det_3, self).__init__()
        
        input_dim = 24
        
        conv0_kernel_size = 3
        conv1_kernel_size = 6
        conv2_kernel_size = 3
        
        lin0_input_size = input_dim
        lin0_output_size = round(input_dim * 2.5)
        
        lin1_input_size = lin0_output_size - conv1_kernel_size - conv2_kernel_size + 2
        lin1_output_size = input_dim
        
        lin2_input_size = lin1_output_size
        lin2_output_size = 12
        
        lin3_input_size = lin2_output_size
        lin3_output_size = 1
        
        self.lin0 = torch.nn.Linear(lin0_input_size, lin0_output_size)
        self.lin1 = torch.nn.Linear(lin1_input_size, lin1_output_size)
        self.lin2 = torch.nn.Linear(lin2_input_size, lin2_output_size)
        self.lin3 = torch.nn.Linear(lin3_input_size, lin3_output_size)
        
        self.conv0 = torch.nn.Conv1d(1, 1, conv0_kernel_size, padding = math.floor(conv0_kernel_size / 2))
        self.conv1 = torch.nn.Conv1d(1, 1, conv1_kernel_size)
        self.conv2 = torch.nn.Conv1d(1, 1, conv2_kernel_size)
    
    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.lin0(x))
        x = F.relu(F.dropout(self.conv1(x), p = 0.25))
        x = F.relu(F.dropout(self.conv2(x), p = 0.2))
        x = F.dropout(F.relu(self.lin1(x)), p = 0.1)
        x = F.dropout(F.relu(self.lin2(x)), p = 0.1)
        x = F.sigmoid(self.lin3(x))
        
        return x

def check_if_valid_beat(slice_range, beats, tolerance = 0):
    """
    Funcion para chequear si el rango pasado se encuentra en alguno de los
    latidos pasados.
    
    Parameters
    ----------
    slice_range : tuple
        Tupla indicando el rango a analizar en muestras.
    beats : list
        Lista con los latidos en muestras.
    tolerance : int
        Tolerancia del rango en muestras.
    
    Returns
    ----------
    valid_beat : bool
        True si hay latido en el rango pasado, False de lo contrario.
    """
    if tolerance != 0:
        tolerance = abs(tolerance)
        slice_range = (slice_range[0] - tolerance, slice_range[1] + tolerance)
    
    beats = sorted(b for b in beats if b <= slice_range[1])
    beats = sorted(b for b in beats if b >= max(slice_range[0], 0))
    
    if len(beats) > 0:
        return True
    else:
        return False
    
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