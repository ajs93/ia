#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:02:32 2019

@author: augusto
"""

import os

import wfdb

import math

import numpy as np
from scipy import signal

from fractions import Fraction

def get_recording_and_resample(recording, target_freq, pre_processing = None):
    """
    Lee el recording pedido y lo resamplea, ejecutando la tarea de pre
    procesamiento antes de devolver.
    
    Parameters
    ----------
    recording : str
        Ruta al archivo del recording.
    target_freq : int
        Frecuencia a la cual resamplear el recording.
    pre_processing : ptr
        Funcion de pre-procesamiento luego de leer el recording.
    
    Returns
    ----------
    data : array        
        Datos leidos resampleados y procesados.
    fields : dict
        Header de los datos.
    beats : list
        Anotaciones resampleadas.
    """
    
    # Elimino posible extension del recording
    recording = os.path.splitext(recording)[0]
    
    # Obtengo señal pedida
    data, fields = wfdb.rdsamp(recording)
    annotation = wfdb.rdann(recording, 'atr')
    
    # Pre-procesamiento en caso de ser pasado
    if not pre_processing == None:
        data, fields = pre_processing(data, fields)
    
    # Filtro anotaciones que no sean latidos
    no_beats_keys = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']
    not_beats_mk = np.isin(annotation.symbol, no_beats_keys, assume_unique = True)
    beats_mk = np.logical_and( np.ones_like(not_beats_mk), np.logical_not(not_beats_mk) )
    
    # Me quedo con las posiciones
    beats = annotation.sample[beats_mk]
    
    # Resampleo tanto de señal como de anotaciones
    resample_fraction = Fraction(target_freq, fields['fs'])
    
    data = signal.resample_poly(data, resample_fraction.numerator, resample_fraction.denominator)
    beats = np.round(beats * resample_fraction.numerator / resample_fraction.denominator).astype(int)
    
    
    fields['fs'] = target_freq
    fields['sig_len'] = len(data[:,0])
    
    return data, fields, beats

def check_if_range_is_beat(r, beats):
    """
    TODO: Documentation
    """
    ret = 0
    
    for this_beat in beats:
        if this_beat < r[0]:
            continue
        
        if this_beat > r[1]:
            break
        
        if this_beat >= r[0] and this_beat <= r[1]:
            ret = 1
            break
    
    return ret
    
def make_train_set_rnn(recording, slice_time, target_freq, channels = None, pre_processing = None):
    """
    Toma un recording, lo divide en slices y labels ordenados.
    
    Parameters
    ----------
    recording : str
        Ruta al archivo del recording.
    slice_time : float
        Duracion en segundos del tiempo de division deseado.
    target_freq : int
        Frecuencia a la cual resamplear el recording.
    channels : list
        Si se pasa este parametro, se procesaran unicamente los canales
        indicados en la lista, si es que se encuentran. Si no se encuentra
        ninguno de los canales pedidos en el recording, los parametros
        devueltos seran None.
    pre_processing: ptr
        Funcion de pre-procesamiento previo a generar la division del recording
        en pedazos. Util para eliminar baseline wanders y probar cosas por el
        estilo. Se le pasa la señal entera y el archivo de descripcion devuelto
        por wfdb.rdsamp(...) y debe devolver la señal ya procesada y un
        diccionario con por lo menos los mismos parametros que el pasado,
        aunque su valor pueda haber cambiado (por ejemplo, se resampleo).
    
    Returns
    ----------
    slices : list
        Lista de intervalos de señal
    labels : list
        Lista de labels de igual tamaño que slices
    """
    
    # Lectura de recording y anotaciones resampleadas y pre-procesadas
    data, fields, beats = get_recording_and_resample(recording, target_freq, pre_processing)
    
    channel_names = fields['sig_name']
    
    # Filtro canales deseados en caso de ser necesario
    channel_filter = []
    
    if not channels == None:
        for counter in range(len(channel_names)):
            if channel_names[counter] in channels:
                channel_filter.append(counter)
        
        if len(channel_filter) == 0:
            # No hay ningun canal con el filtro pasado
            return None, None
        else:
            # Elimino los canales innecesarios
            data = data[:,channel_filter]
            
            fields['sig_name'] = [channel_names[i] for i in channel_filter]
            fields['units'] = [fields['units'][i] for i in channel_filter]
            fields['n_sig'] = len(channel_filter)
            fields['beats'] = beats
    
    # Tamaño de slice en muestras
    slice_in_samples = math.floor(slice_time * target_freq)
    
    # Creacion de lista de señal
    slices = [data[i:i + slice_in_samples - 1] for i in range(0, len(data), slice_in_samples)]
    
    # Elimino el ultimo slice porque puede no llegar a ser del tamaño deseado
    del slices[-1]
    
    # Creacion de los labels
    labels = [check_if_range_is_beat((i, i + slice_in_samples), beats) for i in range(0, len(data), slice_in_samples)]
    
    # Elimino el ultimo label porque por la misma razon que con el slice
    del labels[-1]
    
    return slices, labels