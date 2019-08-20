#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:57:40 2019

@author: augusto
"""

import pickle

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pickle.load(open('/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_beta/mitdb/100.bin','rb'))
    
    parte = data['beat_slices'][25]
    
    t = np.arange(0, parte[1]-parte[0])
    
    t = t / 250
    
    t = t * 1000
    
    for this_part in data['beat_slices']:
        plt.plot(data['data'][this_part[0]:this_part[1]])
        
        axes = plt.gca()
        axes.set_ylim([-1,1])
        axes.set_xlabel('t [mseg]')
        
        plt.show()