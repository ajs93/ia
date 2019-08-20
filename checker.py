#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:03:46 2019

@author: augusto
"""

import os
import pickle

import qrs_detector

import torch

if __name__ == "__main__":
    # Forma de chequeo
    realistic_check = False
    """
    Si la variable de arriba es False, va a tomar absolutamente todos los
    slices en cada registro y calcular la matriz de confusion y sus respectivos
    indicadores en base a los resultados en todos los slices previamente
    generados. Si la misma es True, va a recortar cada recording en slices de
    tamaño slice_samples y va a evaluar en los mismos, como si fuese un 
    recording real (esto va a tardar mucho menos que lo anterior).
    """
    
    # Cantidad de muestras en cada slice
    slice_samples = 24
    
    # Ruta y nombre del modelo guardado
    ia_file_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia"
    ia_filename = "qrs_det_model_3_nofilter.pt"
    
    # Archivos contra los que testear
    test_path = "/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_nofilter"
    rec_path = test_path
    test_filename = "test_set.txt"
    
    # Archivos a testear
    test_file_handler = open(os.path.join(test_path, test_filename), 'r')
    test_files = test_file_handler.read()
    test_files = test_files.split('\n')
    test_files = list(filter(None, test_files))
    test_files = [os.path.join(test_path, s) for s in test_files]
    test_file_handler.close()
    
    # Modelo de IA a testear
    model = qrs_detector.qrs_det_3()
    model.load_state_dict(torch.load(os.path.join(ia_file_path, ia_filename)))
    model.eval()
    
    # Matriz de confusion
    conf_matrix = []
    
    print("Entering test loop...")
    
    if realistic_check:
        print("Mode: Realistic check.")
    else:
        print("Mode: Full check.")
    
    for this_file_idx, this_file in enumerate(test_files):
        print("Evaluating file {}/{}".format(this_file_idx + 1, len(test_files)))
        this_conf_matrix = {}
        this_conf_matrix['file_name'] = this_file
        this_conf_matrix['TP'] = 0
        this_conf_matrix['FP'] = 0
        this_conf_matrix['TN'] = 0
        this_conf_matrix['FN'] = 0
        
        # Cargo datos
        data = pickle.load(open(this_file, 'rb'))
        
        if realistic_check:
            # Separo datos en pedazos
            sign = [data['data'][i:i + slice_samples] for i in range(0, len(data['data']), slice_samples)]
            
            for this_slice_idx, this_slice in enumerate(sign):
                #print("Evaluating index {}/{}".format(this_slice_idx + 1, len(sign)))
                
                # Si no pongo este if, el ultimo pedazo podria ser de distinto tamaño y fallar
                if len(this_slice) == slice_samples:
                    # Analisis
                    this_data = torch.Tensor(this_slice)
                    this_data = this_data.view(1, 1, slice_samples) # Reshape
                    y_pred = model(this_data) # Evaluacion
                    
                    # Dependiendo de lo evaluado, actualizo la matriz de confusion
                    if y_pred.item() >= 0.5:
                        # La IA interpreto que hay latido
                        if qrs_detector.check_if_valid_beat((this_slice_idx * slice_samples, this_slice_idx * slice_samples + slice_samples),
                                               data['fields']['beats'],
                                               round(slice_samples/2)):
                            # Habia latido, TP
                            this_conf_matrix['TP'] += 1
                        else:
                            # No habia latido, FN
                            this_conf_matrix['FN'] += 1
                    else:
                        # La IA interpreto que no hay latido
                        if qrs_detector.check_if_valid_beat((this_slice_idx * slice_samples, this_slice_idx * slice_samples + slice_samples),
                                               data['fields']['beats'],
                                               0):
                            # Habia latido, FP
                            this_conf_matrix['FP'] += 1
                        else:
                            # No habia latido, TN
                            this_conf_matrix['TN'] += 1
            
        else:
            for this_slice_idx, this_slice in enumerate(data['beat_slices']):
                if this_slice_idx % (round(len(data['beat_slices']) / 20)) == 0:
                    print("{}/{} beat slices evaluated.".format(this_slice_idx, len(data['beat_slices'])))
                
                this_data = torch.Tensor(data['data'][this_slice[0]:this_slice[1]])
                this_data = this_data.view(1, 1, slice_samples) # Reshape
                y_pred = model(this_data) # Evaluacion
                
                if y_pred.item() >= 0.5:
                    # La IA interpreto que hay latido, TP
                    this_conf_matrix['TP'] += 1
                else:
                    # La IA interpreto que no hay latido, FN
                    this_conf_matrix['FN'] += 1
            
            for this_slice_idx, this_slice in enumerate(data['no_beat_slices']):
                if this_slice_idx % (round(len(data['no_beat_slices']) / 20)) == 0:
                    print("{}/{} no beat slices evaluated.".format(this_slice_idx, len(data['no_beat_slices'])))
                
                this_data = torch.Tensor(data['data'][this_slice[0]:this_slice[1]])
                this_data = this_data.view(1, 1, slice_samples) # Reshape
                y_pred = model(this_data) # Evaluacion
                
                if y_pred.item() >= 0.5:
                    # La IA interpreto que hay latido, FP
                    this_conf_matrix['FP'] += 1
                else:
                    # La IA interpreto que no hay latido, TN
                    this_conf_matrix['TN'] += 1
        
        # Termino de armar la matriz de confusion, imprimo resultados
        specifiers = qrs_detector.make_specifiers(this_conf_matrix)
        
        print("F1: {}".format(specifiers['F1']))
        print("ACC: {}".format(specifiers['ACC']))
        print("MCC: {}".format(specifiers['MCC']))
        print("--------------------------------------------------")
        
        conf_matrix.append(this_conf_matrix)
    
    # Termino todos los archivos, hacer lo que pinte
    