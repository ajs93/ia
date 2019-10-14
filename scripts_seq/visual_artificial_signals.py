#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:15:12 2019

@author: augusto
"""

import os
import pickle

import torch

import qrs_detector

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

def generate_artificial_signals(sample_freq, signal_freq, stop_time = 10):
    """
    Genera cuadrada triangualar y senoidal a la frecuencia de sampleo deseada.
    
    Parameters
    ----------
    sample_freq : int
        Frecuencia de sampleo deseada.
    signal_freq : int
        Frecuencia de la señal deseada.
    stop_time : int
        Largo del vector en segundos.
    
    Returns
    ----------
    data : dict
        Diccionario con los campos "sine" "square" y "triang".
    """
    
    # Vector temporal
    t = np.linspace(0, 1, stop_time * sample_freq)
    
    # Generacion de señales
    data = {}
    
    data['sine'] = np.sin(t * 2 * np.pi * signal_freq)
    data['triang'] = signal.sawtooth(t * 2 * np.pi * signal_freq, 0.5)
    data['square'] = signal.square(t * 2 * np.pi * signal_freq)
    
    return data

if __name__ == "__main__":
    slice_samples = 24
    
    # Ruta y nombre del modelo guardado
    ia_file_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_1_beta_large"
    ia_filename = "qrs_det_model_1_epoch_10.pt"
    
    # Señales a testear
    signal_freq = 50
    
    data = generate_artificial_signals(250, signal_freq, stop_time = 10)
    
    # Cargo modelo
    model = qrs_detector.qrs_det_1_beta(24)
    model.load_state_dict(torch.load(os.path.join(ia_file_path, ia_filename)))
    model.eval()
    
    counter = 0
    
    while True:
        this_square_data = torch.Tensor(data['square'][counter * slice_samples:counter * slice_samples + slice_samples])
        this_sin_data = torch.Tensor(data['sine'][counter * slice_samples:counter * slice_samples + slice_samples])
        this_triang_data = torch.Tensor(data['triang'][counter * slice_samples:counter * slice_samples + slice_samples])
        
        this_square_data = this_square_data.view(1, 1, 24)
        this_sin_data = this_sin_data.view(1, 1, 24)
        this_triang_data = this_triang_data.view(1, 1, 24)
        
        y_pred_square = model(this_square_data)
        y_pred_sin = model(this_sin_data)
        y_pred_triang = model(this_triang_data)
        
        print("Valor predecido cuadrada: ", y_pred_square.item())
        print("Valor predecido senoidal: ", y_pred_sin.item())
        print("Valor predecido triangular: ", y_pred_triang.item())
        
        if y_pred_square.item() >= 0.5:
            graph_color_square = 'b'
        else:
            graph_color_square = 'r'
        
        if y_pred_sin.item() >= 0.5:
            graph_color_sin = 'b'
        else:
            graph_color_sin = 'r'
        
        if y_pred_triang.item() >= 0.5:
            graph_color_triang = 'b'
        else:
            graph_color_triang = 'r'
        
        print("----------------------------------------------")
        
        plt.subplot(3,1,1)
        plt.plot(this_square_data[0][0].tolist(), graph_color_square)
        plt.grid(True)
        plt.ylabel("Cuadrada")
        
        plt.subplot(3,1,2)
        plt.plot(this_sin_data[0][0].tolist(), graph_color_sin)
        plt.grid(True)
        plt.ylabel("Senoidal")
        
        plt.subplot(3,1,3)
        plt.plot(this_triang_data[0][0].tolist(), graph_color_triang)
        plt.grid(True)
        plt.ylabel("Triangular")
        
        plt.show()
        
        counter += 1
        if counter * slice_samples + slice_samples >= len(data['square']):
            break