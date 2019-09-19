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

import matplotlib.pyplot as plt

if __name__ == "__main__":
    slice_samples = 24
    
    # Ruta y nombre del modelo guardado
    ia_file_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_1_large"
    ia_filename = "qrs_det_model_1_epoch_10.pt"
    
    # Archivo a testear
    rec_path = "/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_agosto/mitdb"
    rec_filename = "217.bin"
    
    # Cargo modelo
    model = qrs_detector.qrs_det_1_beta(24)
    model.load_state_dict(torch.load(os.path.join(ia_file_path, ia_filename)))
    model.eval()
    
    # Cargo archivo a testear
    data = pickle.load(open(os.path.join(rec_path, rec_filename), 'rb'))
    
    counter = 0
    
    while True:
        this_data = torch.Tensor(data['data'][counter * slice_samples:counter * slice_samples + slice_samples])
        this_data = this_data.view(1, 1, 24)
        
        y_pred = model(this_data)
        
        print("Valor predecido: ", y_pred.item())
        
        if y_pred.item() >= 0.5:
            if qrs_detector.check_if_valid_beat((counter * slice_samples, counter * slice_samples + slice_samples),
                                            data['fields']['beats'],
                                            round(slice_samples/2)):
                print("El checkeador dice que es verdadero.")
            else:
                print("El checkeador dice que es falso.")
            
            graph_color = 'b'
        else:
            if qrs_detector.check_if_valid_beat((counter * slice_samples, counter * slice_samples + slice_samples),
                                            data['fields']['beats'],
                                            round(0)):
                print("El checkeador dice que es verdadero.")
            else:
                print("El checkeador dice que es falso.")
            
            graph_color = 'r'
        
        print("----------------------------------------------")
        
        plt.plot(this_data[0][0].tolist(), graph_color)
        
        axes = plt.gca()
        axes.set_ylim([-3,3])
        
        plt.show()
        
        counter += 1
        if counter * slice_samples + slice_samples >= data['fields']['sig_len']:
            break