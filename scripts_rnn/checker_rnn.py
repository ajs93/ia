#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:03:46 2019

@author: augusto
"""

import os
import pickle
import statistics

import qrs_detector

import torch

from prettytable import PrettyTable

def make_table(results):
    table = PrettyTable()
    
    table.field_names = [" ", "F1", "ACC", "MCC"]
    
    F1_list = []
    ACC_list = []
    MCC_list = []
    
    results['file_name'] = sorted(results['file_name'])
    
    for (this_specs, this_file) in zip(results['specifiers'], results['file_name']):
        this_row = []
        this_row.append(this_file)
        this_row.append("{:1.2f} %".format(this_specs['F1'] * 100))
        this_row.append("{:1.2f} %".format(this_specs['ACC'] * 100))
        this_row.append("{:1.2f} %".format((this_specs['MCC'] + 1) / 2 * 100))
        
        F1_list.append(this_specs['F1'])
        ACC_list.append(this_specs['ACC'])
        MCC_list.append(this_specs['MCC'])
        
        table.add_row(this_row)
    
    table.add_row(["median",
                   "{:1.2f} %".format(statistics.median(F1_list) * 100),
                   "{:1.2f} %".format(statistics.median(ACC_list) * 100),
                   "{:1.2f} %".format((statistics.median(MCC_list) + 1) / 2 * 100)])
    
    table.add_row(["mean",
                   "{:1.2f} %".format(statistics.mean(F1_list) * 100),
                   "{:1.2f} %".format(statistics.mean(ACC_list) * 100),
                   "{:1.2f} %".format((statistics.mean(MCC_list) + 1) / 2 * 100)])
    
    return table

if __name__ == "__main__":
    # Cantidad de muestras en cada slice
    slice_samples = 24
    
    # Ruta y nombre del modelo guardado
    ia_file_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_rnn"
    ia_filename = "qrs_det_model_rnn.pt"
    
    # Archivos contra los que testear
    test_path = "/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_rnn"
    rec_path = test_path
    test_filename = "validation_set.txt"
    
    # Ruta y nombre donde guardar los resultados
    save_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_rnn"
    save_filename = "qrs_det_model_rnn_results.bin"
    
    if not os.path.isfile(os.path.join(save_path, save_filename)):
        # Archivos a testear
        test_file_handler = open(os.path.join(test_path, test_filename), 'r')
        test_files = test_file_handler.read()
        test_files = test_files.split('\n')
        test_files = list(filter(None, test_files))
        test_filepaths = [os.path.join(test_path, s) for s in test_files]
        test_file_handler.close()
        
        # Modelo de IA a testear
        model = qrs_detector.qrs_det_rnn(slice_samples)
        model.load_state_dict(torch.load(os.path.join(ia_file_path, ia_filename)))
        model.eval()
        
        # Matriz de confusion
        results = {}
        results['confusion_matrix'] = []
        results['specifiers'] = []
        results['file_name'] = []
        results['file_path'] = []
        
        print("Entering test loop...")
        
        for this_file_idx, this_file in enumerate(test_filepaths):
            print("Evaluating file {}/{}".format(this_file_idx + 1, len(test_filepaths)))
            this_conf_matrix = {}
            this_conf_matrix['TP'] = 0
            this_conf_matrix['FP'] = 0
            this_conf_matrix['TN'] = 0
            this_conf_matrix['FN'] = 0
            
            # Cargo datos
            data = pickle.load(open(this_file, 'rb'))
            
            # Inicializo hidden state de la red
            hidden_state = None
            
            for this_slice_idx, (this_slice, this_label) in enumerate(zip(data['slices'], data['labels'])):
                if this_slice_idx % (round(len(data['slices']) / 20)) == 0:
                    print("{}/{} slices evaluated.".format(this_slice_idx, len(data['slices'])))
                
                this_data = torch.Tensor(this_slice)
                this_data = this_data.view(1, 1, slice_samples) # Reshape
                y_pred, hidden_state = model(this_data, hidden_state) # Evaluacion
                
                if y_pred.item() >= 0.5:
                    # La IA Interpreto que hay latido
                    if this_label == 1:
                        # True positive
                        this_conf_matrix['TP'] += 1
                    else:
                        # False positive
                        this_conf_matrix['FP'] += 1
                else:
                    if this_label == 0:
                        # True negative
                        this_conf_matrix['TN'] += 1
                    else:
                        # False negative
                        this_conf_matrix['FN'] += 1
            
            # Termino de armar la matriz de confusion, imprimo resultados
            this_specifiers = qrs_detector.make_specifiers(this_conf_matrix)
            
            print("F1: {}".format(this_specifiers['F1']))
            print("ACC: {}".format(this_specifiers['ACC']))
            print("MCC: {}".format(this_specifiers['MCC']))
            print("--------------------------------------------------")
            
            results['confusion_matrix'].append(this_conf_matrix)
            results['specifiers'].append(this_specifiers)
            results['file_name'].append(test_files[this_file_idx])
            results['file_path'].append(this_file)
        
        # Hago la tabla con los resultados obtenidos
        table = make_table(results)
        
        print(table)
    else:
        results = pickle.load(open(os.path.join(save_path, save_filename), 'rb'))
        
        table = make_table(results)
        
        print(table)
    
    # Termino todos los archivos, guardo el analisis
    pickle.dump(results, open(os.path.join(save_path, save_filename), 'wb'))