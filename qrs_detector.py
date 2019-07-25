#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:05:41 2019

@author: augusto
"""

import os
import pickle
import random

import numpy as np

from keras import backend as K

from keras.utils import Sequence

from keras.models import Sequential

from keras.layers import Conv1D, BatchNormalization, GlobalMaxPooling1D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import AveragePooling1D

from keras.optimizers import Adam

class my_generator(Sequence):
    """
    Clase generadora para el entrenamiento de la IA.
    Por ahora esta hecha para que en cada batch tome un archivo distinto.
    """
    def __init__(self, database_source_path, target_file_path, batch_size = 32, beats_rel = 0.7):
        """
        Parameters
        ----------
        database_source_path : str
            Ruta al directorio donde se encuentran las muestras.
        target_file_path : str
            Ruta al archivo de donde se sacaran las muestras para el
            entrenamiento.
        batch_size : int
            Tamaño del batch size de cada vuelta.
        beats_rel : float
            Porcentaje respecto al 100% del batch que se devuelven latidos.
        """
        self.batch_size = batch_size
        self.beats_rel = beats_rel
        
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
            
            aux_slice_counter = np.floor(len(this_data['beat_slices'])/self.batch_size).astype(int)
            aux_slice_counter += np.floor(len(this_data['no_beat_slices'])/self.batch_size).astype(int)
            
            self.slice_limits.append(aux_slice_counter)
            self.total_slices += aux_slice_counter
            
            this_file.close()
        
    def __len__(self):
        """
        Tiene que devolver la cantidad de batches que hay en los datos.
        """
        return np.floor(self.total_slices / self.batch_size).astype(int)
    
    def __getitem__(self, idx):
        """
        Tiene que devolver un batch de muestras y labels.
        """
        # Obtengo indice interno
        real_idx = 0
        
        while idx > self.slice_limits[real_idx]:
            real_idx += 1
        
        # Obtengo datos del idx pedido
        this_data = pickle.load(open(self.file_names[int(real_idx)], 'rb'))
        
        # Esto habria que chequear si es necesario:
        """
        this_data['beat_slices'] = this_data['beat_slices'][0:-1]
        this_data['no_beat_slices'] = this_data['no_beat_slices'][0:-1]
        """
        
        # Dependiendo del batch size, siempre devuelvo una relacion de latidos a no latidos
        beats_in_batch = int(np.round(self.beats_rel * self.batch_size))
        no_beats_in_batch = self.batch_size - beats_in_batch
        
        beats_indexes = random.sample(range(len(this_data['beat_slices']) - 1), beats_in_batch)
        no_beats_indexes = random.sample(range(len(this_data['no_beat_slices']) - 1), no_beats_in_batch)
        
        # Con los indices, obtengo los datos necesarios
        batch_x = []
        batch_y = []
        
        for counter in beats_indexes:
            this_start = this_data['beat_slices'][counter][0]
            this_end = this_data['beat_slices'][counter][1]
            
            batch_x.append(this_data['data'][this_start:this_end])
            batch_y.append(1)
        
        for counter in no_beats_indexes:
            this_start = this_data['no_beat_slices'][counter][0]
            this_end = this_data['no_beat_slices'][counter][1]
            
            batch_x.append(this_data['data'][this_start:this_end])
            batch_y.append(0)
        
        return_x = np.array(batch_x)
        return_y = np.array(batch_y)
        
        return return_x, return_y

def t_se(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def t_pp(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def t_f1(y_true, y_pred):
    precision = t_pp(y_true, y_pred)
    recall = t_se(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if __name__ == "__main__":
    # Tamaño de cada batch
    batch_size = 512
    beats_rel = 0.7 # 70% del batch latidos, 30% no latidos
    total_epochs = 5
    
    # Creacion de generadores tanto de entrenamiento como validacion
    train_gen = my_generator('/mnt/Databases/processed/only_MLII/', '/mnt/Databases/processed/only_MLII/train_set.txt', batch_size = batch_size, beats_rel = beats_rel)
    val_gen = my_generator('/mnt/Databases/processed/only_MLII/', '/mnt/Databases/processed/only_MLII/validation_set.txt', batch_size = batch_size, beats_rel = beats_rel)
    
    # Definicion del modelo
    model = Sequential()
    
    model.add(Conv1D(128,
                     16,
                     activation = None,
                     input_shape = train_gen.shape,
                     padding = 'valid'))
    
    model.add(BatchNormalization())
    
    model.add(Conv1D(128,
                     16,
                     activation = 'relu',
                     padding = 'valid'))
    
    model.add(BatchNormalization())
    
    model.add(GlobalMaxPooling1D())
    
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(25))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(1))
    
    model.add(Activation('sigmoid'))
    
    # Compilacion del modelo
    model.compile(loss='binary_crossentropy',
                  optimizer = Adam(),
                  metrics = [t_f1, t_pp, t_se])
    
    # Imprimo caracteristicas del modelo
    print(model.summary())
    
    # Entrenamiento del modelo
    
    history = model.fit_generator(train_gen,
                                  use_multiprocessing = True,
                                  epochs = total_epochs,
                                  validation_data = val_gen)