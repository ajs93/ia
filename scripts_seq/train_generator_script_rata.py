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
    butt_order_linea = 4
    
    freq_linea_hz = 50
    freq_linea = freq_linea_hz * 2 / fields['fs']
    
    # Pasa altos en freq_corte_hz para remover continua
    b, a = signal.butter(np.round(butt_order / 2).astype(int), freq_corte, 'hp')
    data = signal.filtfilt(b, a, data, axis = 0)
    
    # Filtro notch para remover el ruido de linea en 50Hz
    b, a = signal.butter(np.round(butt_order_linea / 2).astype(int), [freq_linea * 0.95, freq_linea * 1.05], 'bandstop')
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
    databases = ['ratadb_segmentada_ia']
    
    # Destino a donde guardar los datos procesados
    processed_files_directory = '/home/augusto/Desktop/GIBIO/processed_dbs/rata_segmentada_ia_v3'
    
    ext_filter = '.mat'
    
    if not processed_files_directory[-1] == os.sep:
        processed_files_directory +=  os.sep
    
    if not os.path.exists(processed_files_directory):
        os.makedirs(processed_files_directory)
    
    # Archivos para guardar la division de recordings
    train_output_file = open(processed_files_directory + 'train_set.txt', 'w')
    validation_output_file = open(processed_files_directory + 'validation_set.txt', 'w')
    test_output_file = open(processed_files_directory + 'test_set.txt', 'w')
    
    # Parametros para la preparacion de los datos
    target_freq = 1000 # Frecuencia objetivo de resampling
    slice_time = 25e-3 # Tiempo en segundos de las ventanas de slice
    valid_window_time = 0 # Margen de tolerancia de la anotacion
    channels = ['  Dev1_ai1']
    
    input('Press enter to start.')
    
    for this_database in databases:
        print('Processing database: ' + this_database)
        
        # Todos los recordings de la base de datos
        recording_names = os.listdir(database_directory + this_database)
        
        # Filtro
        recording_names = [s for s in recording_names if ext_filter in s]
        
        # Mezclo al azar los recordings
        random.shuffle(recording_names)
        
        record_counter = 0
        
        # Genero datos necesarios
        for this_record in recording_names:
            print('Processing record: ' + this_record + '...')
            
            data = {}
            
            data['beat_slices'], data['no_beat_slices'], data['data'], data['fields'] = data_generation_utils.make_train_set(database_directory + this_database + os.sep + this_record,
                                                                  slice_time,
                                                                  valid_window_time,
                                                                  target_freq,
                                                                  return_intervals = True,
                                                                  channels = channels,
                                                                  pre_processing = pre_function)
            
            # Puede que en este recording no haya ningun canal deseado
            if not data['data'] is None:
                # Debug. Me quedo con la un tercio de latidos que no latidos
                cant_latidos = len(data['beat_slices'])
                data['no_beat_slices'] = random.sample(data['no_beat_slices'], min(round(cant_latidos * 3), len(data['no_beat_slices'])))
                
                # Guardo datos en archivo
                output_file_name = processed_files_directory + this_database + os.sep + this_record[0:-4] + '.bin'
                
                # Creacion del directorio en caso que no exista
                if not os.path.exists(processed_files_directory + this_database):
                    os.makedirs(processed_files_directory + this_database)
                
                # Guardado de datos
                output_file = open(output_file_name, 'wb')
                
                pickle.dump(data, output_file)
                
                output_file.close()
                
                if record_counter <= 1:
                    train_output_file.write(this_database + os.sep + this_record[0:-4] + '.bin\n')
                elif record_counter == 2:
                    validation_output_file.write(this_database + os.sep + this_record[0:-4] + '.bin\n')
                else:
                    test_output_file.write(this_database + os.sep + this_record[0:-4] + '.bin\n')
                
                record_counter += 1
                record_counter %= 4
                
                print('Recording ' + this_record + ' processed succesfully.')
            else:
                print('Recording ' + this_record + ' does not have desired channel/s.')
    
    train_output_file.close()
    validation_output_file.close()
    test_output_file.close()
    
    print('Ended processing.')