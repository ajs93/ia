#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:09:19 2019

@author: augusto
"""

import data_generation_utils

import os
import pickle

import numpy as np

from scipy import signal

import random

def pre_function(data, fields):
    freq_corte_hz = 2
    freq_corte = freq_corte_hz * 2 / fields['fs']
    butt_order = 2
    
    # Pasa altos en freq_corte_hz para remover continua
    b, a = signal.butter(np.round(butt_order / 2).astype(int), freq_corte, 'hp')
    data = signal.filtfilt(b, a, data, axis = 0)
    
    return data, fields

if __name__ == "__main__":
    """
    Script para la generacion de datos para el entrenamiento de la IA.
    
    La relacion (no_latidos)/(latidos) sera 3 a 1.
    
    La cantidad total de datos, se separara en 3 sets:
    Entrenamiento = 50%
    Validacion = 25%
    Test  25%
    """
    
    # Bases de datos a procesar
    database_directory = '/home/augusto/Desktop/GIBIO/Databases/'
    databases = ['mitdb', 'INCART']
    
    # Destino a donde guardar los datos procesados
    processed_files_directory = '/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_rnn'
    
    if not processed_files_directory[-1] == os.sep:
        processed_files_directory +=  os.sep
    
    if not os.path.exists(processed_files_directory):
        os.makedirs(processed_files_directory)
    
    # Archivos para guardar la division de recordings
    train_output_file = open(processed_files_directory + 'train_set.txt', 'w')
    validation_output_file = open(processed_files_directory + 'validation_set.txt', 'w')
    test_output_file = open(processed_files_directory + 'test_set.txt', 'w')
    
    # Parametros para la preparacion de los datos
    target_freq = 250 # Frecuencia objetivo de resampling
    slice_time = 100e-3 # Tiempo en segundos de las ventanas de slice
    channels = ['MLII', 'II', 'ML2']
    
    for this_database in databases:
        print('Processing database: ' + this_database)
        
        # Todos los recordings de la base de datos
        recording_names = os.listdir(database_directory + this_database)
        
        # Filtro
        recording_names = [s for s in recording_names if '.dat' in s]
        
        # Elimino extension
        recording_names = [os.path.splitext(s)[0] for s in recording_names]
        
        # Mezclo al azar los recordings
        random.shuffle(recording_names)
        
        record_counter = 0
        
        # Genero datos necesarios
        for this_record in recording_names:
            print('Processing record: ' + this_record + '...')
            
            data = {}
            
            this_file = os.path.join(database_directory + this_database, this_record)
            
            data['slices'], data['labels'] = data_generation_utils.make_train_set_rnn(this_file,
                                                              slice_time,
                                                              target_freq,
                                                              channels = channels,
                                                              pre_processing = None)
            
            # Puede que en este recording no haya ningun canal deseado
            if not data['slices'] is None:
                # Guardo datos en archivo
                output_file_name = os.path.join(processed_files_directory + this_database, this_record + '.bin')
                
                # Creacion del directorio en caso que no exista
                if not os.path.exists(processed_files_directory + this_database):
                    os.makedirs(processed_files_directory + this_database)
                
                # Guardado de datos
                output_file = open(output_file_name, 'wb')
                
                pickle.dump(data, output_file)
                
                output_file.close()
                
                if record_counter <= 1:
                    train_output_file.write(this_database + os.sep + this_record + '.bin\n')
                elif record_counter == 2:
                    validation_output_file.write(this_database + os.sep + this_record + '.bin\n')
                else:
                    test_output_file.write(this_database + os.sep + this_record + '.bin\n')
                
                record_counter += 1
                record_counter %= 4
                
                print('Recording ' + this_record + ' processed succesfully.')
            else:
                print('Recording ' + this_record + ' does not have desired channel/s.')
    
    train_output_file.close()
    validation_output_file.close()
    test_output_file.close()
    
    print('Ended processing.')