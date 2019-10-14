#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:41:35 2019

@author: augusto
"""

import os
import pickle

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Destination for the synthetically created files
    source_folder = "/home/augusto/Desktop/GIBIO/Databases/synthetic_rata_db/"
    
    # File name pattern
    file_name = "synthetic_rata_1.bin"
    
    whole_data = pickle.load(open(source_folder + file_name, 'rb'))
    
    beats = whole_data['beats']
    data = whole_data['data']
    
    fields = {}
    fields['fs'] = whole_data['fs']
    fields['n_sig'] = 1
    fields['sig_len'] = len(data)
    fields['sig_name'] = ['synth']
    
    t = list(range(len(data)))
    t = [a / fields['fs'] for a in t]
    
    plt.figure()
    plt.plot(t, data)
    plt.plot([a / fields['fs'] for a in beats], [data[b] for b in beats], '*')
    plt.show()