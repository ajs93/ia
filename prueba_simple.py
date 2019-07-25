#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 23:37:24 2019

@author: augusto
"""

import pickle

if __name__ == "__main__":
    archivo_1 = '/mnt/Databases/processed/only_MLII/mitdb/100.bin'
    archivo_2 = '/mnt/Databases/processed/only_MLII/INCART/I02.bin'
    
    data_1 = pickle.load(open(archivo_1,'rb'))
    data_2 = pickle.load(open(archivo_2,'rb'))