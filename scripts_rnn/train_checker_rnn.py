#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:36:50 2019

@author: augusto
"""

import pickle

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pickle.load(open('/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_rnn/mitdb/100.bin', 'rb'))
    
    for this_slice, this_label in zip(data['slices'], data['labels']):
        if this_label == 0:
            color = 'r'
        else:
            color = 'b'
        
        plt.plot(this_slice, color)
        
        axes = plt.gca()
        axes.set_ylim([-1,1])
        
        plt.show()