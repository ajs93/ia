#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 01:39:43 2019

@author: augusto
"""

import qrs_detector

if __name__ == "__main__":
    # Creacion de generador de entrenamiento
    train_gen = qrs_detector.dataset_loader('/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_beta',
                             '/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_beta/train_set.txt')
    
    for idx, item in enumerate(train_gen):
        print("idx:{}/{}".format(idx + 1, train_gen.__len__()))