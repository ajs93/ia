#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:53:30 2019

@author: augusto
"""

import random
import os
import pickle

if __name__ == "__main__":
    noise_max_amplitude = 5
    
    # Destination for the synthetically created files
    destination_folder = "/home/augusto/Desktop/GIBIO/Databases/synthetic_rata_db/"
    
    # File name pattern
    file_name = "synthetic_rata_{}.bin"
    
    # Pattern to use
    pattern_file = "/home/augusto/Desktop/GIBIO/Databases/synthetic_rata_db/pattern{}.pat"
    max_patterns = 3
    
    # Recording characteristics
    rec_fs = 1000 # Hz
    rec_len = 120 # segs
    rec_amount = 15 # How many recordings to synthetize
    
    # Minimum and maximum time difference between two patterns
    min_diff = 100e-3 # segs
    max_diff = 400e-3 # segs
    
    # Start creating
    for rec_count in range(rec_amount):
        print("Generating random signal number {}".format(rec_count + 1))
        
        this_signal = []
        this_beats = []
        samples_counter = 0
        
        while samples_counter < int(rec_len * rec_fs):
            # Define a random length zero time (or zero + noise)
            zero_length = random.randint(rec_fs * min_diff, rec_fs * max_diff)
            
            if noise_max_amplitude == 0:
                this_signal += ([0] * zero_length)
            else:
                this_noise = [random.randint(-noise_max_amplitude, noise_max_amplitude) for _ in range(zero_length)]
                this_signal += this_noise
            
            samples_counter += zero_length
            
            # Then add the pattern, and mark the beat
            this_random_pattern = random.randint(0, max_patterns - 1)
            
            # Load the pattern
            pattern = pickle.load(open(pattern_file.format(this_random_pattern), 'rb'))
            
            this_signal += (pattern)
            this_beats.append(samples_counter)
            samples_counter += len(pattern)
        
        # At this point the synthetic signal was generated, save it
        fields = {}
        
        fields['fs'] = rec_fs
        fields['data'] = this_signal
        fields['beats'] = this_beats
        
        pickle.dump(fields, open(os.path.join(destination_folder, file_name).format(rec_count), 'wb'))