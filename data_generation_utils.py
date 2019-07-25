#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:48:33 2019

@author: Augusto Santini

Funcionalidades para la generacion de los datos
"""

import os

import wfdb

import numpy as np
from scipy import signal

from fractions import Fraction

def my_split(x, split_size):
    """
    Toma un numpy array y lo divide en pedazos de tamaño split_size.
    
    Parameters
    ----------
    x : array
        Datos a dividir.
    split_size : int
        Tamaño deseado de cada pedazo de x.
    
    Returns
    ----------
    y : list
        Lista de numpy arrays divididos.
    """
    total_length = x.shape[0]
    total_splits = int(np.floor(total_length / split_size))
    split_size = int(split_size)
    
    y = []
    
    for i in range(total_splits):
        y.append(x[i*split_size:split_size * (i + 1),:])
    
    return y

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

def make_recording_slices(recording, slice_time, window_size, target_freq, pre_processing = None):
    """
    Toma un recording y lo divide en pedazos de tiempo fijo.
    
    Parameters
    ----------
    recording : str
        Ruta al archivo del recording.
    slice_time : float
        Duracion en segundos del tiempo de division deseado.
    window_size : float
        Ancho de la ventana en segundos donde es valido tomar una secuencia
        como latido.
    target_freq : int
        Frecuencia a la cual resamplear el recording.
    pre_processing : ptr
        Funcion de pre-procesamiento previo a generar la division del recording
        en pedazos. Util para eliminar baseline wanders y probar cosas por el
        estilo. Se le pasa la señal entera y el archivo de descripcion devuelto
        por wfdb.rdsamp(...) y debe devolver la señal ya procesada y un
        diccionario con por lo menos los mismos parametros que el pasado,
        aunque su valor pueda haber cambiado (por ejemplo, se resampleo).
    
    Return
    ----------
    sliced_recording : list
        Lista con slices de señal de tamaño (slice_time * target_freq, n) donde
        n corresponde a la cantidad de canales del recording.
    labels : list
        Lista con booleanos de tamaño (slice_time * target_freq, n) donde n
        corresponde a la cantidad de canales del recording.
    """
    
    # Lectura de recording y anotaciones resampleadas y pre-procesadas
    data, beats, fields = get_recording_and_resample(recording, target_freq, pre_processing)
    
    # Fijo la ventana donde tomaremos como valido que hay latido
    valid_window_in_samples = np.round((window_size / 2) * target_freq)
    
    # Transformacion de tiempo de slice a muestras
    slice_in_samples = slice_time * target_freq
    
    # Generacion de los pedazos de recording
    sliced_recording = my_split(data, slice_in_samples)
    
    labels = []
    
    # Para las anotaciones, queda generar una lista indicando si hay latido o no
    for i in range(len(sliced_recording)):
        this_extremes = [(i * slice_in_samples) - valid_window_in_samples, ((i + 1) * slice_in_samples) + valid_window_in_samples]
        
        if np.where(np.logical_and(beats >= this_extremes[0], beats <= this_extremes[1]))[0].size > 0:
            # Hay latido
            labels.append(True)
        else:
            # No hay latido
            labels.append(False)
    
    return sliced_recording, labels

def get_beats_nobeats_zones(beats, recording_sample_length, freq, slice_time, valid_window_size):
    """
    Toma latidos de un recording y devuelve las zonas de latidos y no latidos.
    
    Parameters
    ----------
    beats : list
        Lista con los valores en muestras de la ocurrencia de los latidos.
    recording_sample_length : int
        Cantidad de muestras en el recording.
    freq : int
        Frecuencia de muestreo.
    slice_time : float
        Cantidad en segundos de la ventana de slice.
    valid_window_size: float
        Ancho de la ventana donde es valido tomar una secuencia como latido, en
        segundos.
    
    Returns
    ----------
    beats_zones : list
        Lista de intervalos (tuplas) donde hay latidos.
    no_beats_zones : list
        Lista de intervalos (tuplas) donde no hay latidos.
    """
    
    freq = int(freq)
    
    # Paso todo a muestras
    slice_time = np.round(slice_time * freq).astype(int)
    valid_window_size = np.round(valid_window_size * freq).astype(int)
    
    # Fuerzo a que la ventana sea de tamaño par
    if slice_time % 2 == 1:
        slice_time -= 1
    
    if valid_window_size % 2 == 1:
        valid_window_size -= 1
    
    # Las ventanas totales las divido en dos
    half_slice_time = np.floor(slice_time / 2).astype(int)
    half_valid_window_size = np.floor(valid_window_size / 2).astype(int)
    
    # Si el valid_window_size era cero, solo se quiere donde sea absolutamente valido el latido
    if valid_window_size == 0:
        valid_window_size = 1
    
    """
    Considerando hacer data augmentation: Para esto, los intervalos donde hay
    latidos, tendran que ser tales que incluyan el momento de ocurrencia del
    latido +/- la ventana de admision de validez. Esto es lo que hago en el
    siguiente doble for.
    """
    beats_zones = []
    
    for this_annotation in beats:
        for this_sample in range(slice_time * 2):
            if this_sample + this_annotation - slice_time >= 0 and this_sample + this_annotation < recording_sample_length - 1:
                beats_zones.append((this_sample + this_annotation - slice_time, this_sample + this_annotation))
    
    """
    # Esto anda, pero tarda:
    for ann_counter in range(len(beats)):
        for slide_counter in range(valid_window_size):
            this_slide = beats[ann_counter] - half_valid_window_size + slide_counter
            
            if this_slide - half_slice_time >= 0 and this_slide + half_slice_time < recording_sample_length:
                beats_zones.append((int(this_slide - half_slice_time), int(this_slide + half_slice_time)))
    """
    
    """
    Lo que queda es las zonas de no latidos. Para esto barro una ventana hasta
    que encuentre algun margen de anotacion.
    """
    no_beats_zones = []
    
    """
    Pajin: Esto lo podes hacer barriendo una lista de tuplas.
    Declarate una lista con los intervalos (cada tupla es un intervalo),
    y barres todo ahi adentro nada mas, te salteas lo demas y te ahorras
    el np.where() que tarda mil años.
    """
    
    no_beat_zones_intervals = []
    
    for this_annotation in range(len(beats)):
        if this_annotation == 0:
            if beats[this_annotation] - half_slice_time >= 0:
                no_beat_zones_intervals.append((this_annotation, beats[this_annotation] - half_slice_time))
        elif this_annotation != len(beats) - 1:
            if beats[this_annotation + 1] - half_slice_time >= 0:
                no_beat_zones_intervals.append((beats[this_annotation] + half_slice_time, beats[this_annotation + 1] - half_slice_time))
        else:
            no_beat_zones_intervals.append((beats[this_annotation] + half_slice_time, recording_sample_length - 1))
    
    for this_interval in no_beat_zones_intervals:
        for this_sample in range(this_interval[1] - this_interval[0] - slice_time):
            no_beats_zones.append((this_sample, this_sample + slice_time))
    
    """
    # Esto anda, pero tarda:
    for slide_counter in range(recording_sample_length - valid_window_size - 1):
        if not np.where(np.logical_and(beats + half_valid_window_size < slide_counter + slice_time, beats - half_valid_window_size > slide_counter))[0].size > 0:
            if int(slide_counter + slice_time) < recording_sample_length - 1:
                no_beats_zones.append((int(slide_counter), int(slide_counter + slice_time)))
    """
    
    return beats_zones, no_beats_zones

def make_train_set(recording, slice_time, valid_window_size, target_freq, channels = None, return_intervals = False, pre_processing = None):
    """
    Toma un recording y lo divide en slices de no latido y slices de latido,
    pudiendo devolver los slices en forma de señal o en forma de intervalos de
    ocurrencia.
    
    Parameters
    ----------
    recording : str
        Ruta al archivo del recording.
    slice_time : float
        Duracion en segundos del tiempo de division deseado.
    valid_window_size : float
        Ancho de la ventana donde es valido tomar una secuencia como latido.
    target_freq : int
        Frecuencia a la cual resamplear el recording.
    channels : list
        Si se pasa este parametro, se procesaran unicamente los canales
        indicados en la lista, si es que se encuentran. Si no se encuentra
        ninguno de los canales pedidos en el recording, los parametros
        devueltos seran None.
    return_intervals : bool
        En caso de querer recuperar unicamente los indices de inicio y fin 
        donde ocurren los latidos/no latidos pasar este parametro como True.
        Caso contrario la funcion devuelve los slices de señal directamente.
    pre_processing: ptr
        Funcion de pre-procesamiento previo a generar la division del recording
        en pedazos. Util para eliminar baseline wanders y probar cosas por el
        estilo. Se le pasa la señal entera y el archivo de descripcion devuelto
        por wfdb.rdsamp(...) y debe devolver la señal ya procesada y un
        diccionario con por lo menos los mismos parametros que el pasado,
        aunque su valor pueda haber cambiado (por ejemplo, se resampleo).
    
    Returns
    ----------
    beat_slices : list
        Si return_intervals == False:
        Lista con slices de señal de tamaño (slice_time * target_freq, n) donde
        n corresponde a la cantidad de canales del recording, donde hay
        latidos.
        Caso contrario:
        Lista con tuplas de inicio y fin donde se encuentran los distintos
        latidos.
    no_beat_slices : list
        Si return_intervals == False:    
        Lista con slices de señal de tamaño (slice_time * target_freq, n) donde
        n corresponde a la cantidad de canales del recording, donde no hay
        latidos.
        Caso contrario:
        Lista con tuplas de inicio y fin donde se encuentran los distintos
        no latidos.
    data : array
        Recording resampleado y procesado.
    fields : dict
        Header de la señal con campos:
            fs: Frecuencia de sampleo
            units: Unidad de cada canal
            comments: Comentario del header del recording
            n_sig: Cantidad de señales
            sig_len: Cantidad de muestras
            sig_name: Nombre de cada canal
            units: Unidades de cada canal
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
            return None, None, None, None
        else:
            # Elimino los canales innecesarios
            data = data[:,channel_filter]
            
            fields['sig_name'] = [channel_names[i] for i in channel_filter]
            fields['units'] = [fields['units'][i] for i in channel_filter]
            fields['n_sig'] = len(channel_filter)
            
    # Obtencion de zonas de interes (latido/no latido)
    beat_zones, no_beat_zones = get_beats_nobeats_zones(beats, data.shape[0], target_freq, slice_time, valid_window_size)
    
    if return_intervals:
        return beat_zones, no_beat_zones, data, fields
    else:
        # Generacion de los slices a partir de las zonas de interes obtenidas
        beat_slices = [data[beat_zones[counter][0]:beat_zones[counter][1],:] for counter in range(len(beat_zones))]
        no_beat_slices = [data[no_beat_zones[counter][0]:no_beat_zones[counter][1],:] for counter in range(len(no_beat_zones))]
        
        return beat_slices, no_beat_slices, data, fields