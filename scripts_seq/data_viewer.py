#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:10:44 2019

@author: augusto
"""

import os
import pickle

import matplotlib.pyplot as plt

if __name__ == "__main__":
    ver_latidos = False
    guardar_pattern = False
    
    pattern_filename = "/home/augusto/Desktop/GIBIO/Databases/synthetic_rata_db/pattern{}.pat"
    
    # Archivo a testear
    rec_path = "/home/augusto/Desktop/GIBIO/processed_dbs/rata_segmentada_ia/ratadb_segmentada_ia"
    rec_filename = "06012006_182543.bin"
    
    # Cargo archivo a testear
    data = pickle.load(open(os.path.join(rec_path, rec_filename), 'rb'))
    
    counter = 0
    
    if guardar_pattern:
        desde = data['beat_slices'][counter][0]
        hasta = data['beat_slices'][counter][1]
        this_data = data['data'][desde:hasta]
        
        plotable_data = this_data.tolist()
        plotable_data = [b[0] for b in plotable_data]
        
        pickle.dump(plotable_data, open(pattern_filename.format(counter), 'wb'))
        
        exit()
    
    while True:
        if ver_latidos:
            desde = data['beat_slices'][counter][0]
            hasta = data['beat_slices'][counter][1]
            this_data = data['data'][desde:hasta]
            graph_color = 'b'
        else:
            desde = data['no_beat_slices'][counter][0]
            hasta = data['no_beat_slices'][counter][1]
            this_data = data['data'][desde:hasta]
            graph_color = 'r'
        
        plotable_data = this_data.tolist()
        plotable_data = [b[0] for b in plotable_data]
        
        plt.figure()
        plt.plot(this_data.tolist(), graph_color)
        
        axes = plt.gca()
        axes.set_ylim([-500,500])
        
        plt.title('counter = {}'.format(counter))
        plt.show()
        
        counter += 1