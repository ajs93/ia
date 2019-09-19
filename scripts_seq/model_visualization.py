#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:17:41 2019

@author: augusto
"""

import os
import pickle
import random

import torch

import qrs_detector

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Ruta y nombre del modelo guardado
    ia_file_path = "/home/augusto/Desktop/GIBIO/Algoritmos/ia/trained_models/model_1_beta_large"
    ia_filename = "qrs_det_model_1_epoch_10.pt"
    
    # Archivo a visualizar
    rec_path = "/home/augusto/Desktop/GIBIO/processed_dbs/only_MLII_agosto/mitdb"
    rec_filename = "103.bin"
    
    model = qrs_detector.qrs_det_1_beta(24)
    model.load_state_dict(torch.load(os.path.join(ia_file_path, ia_filename)))
    model.eval()
    
    # Cargo archivo a testear
    data = pickle.load(open(os.path.join(rec_path, rec_filename), 'rb'))
    
    flag_beat_nobeat = True
    flag_in_order = True
    this_index = 0
    
    desde = data['beat_slices'][10][0]
    hasta = data['beat_slices'][10][1]
    this_data = data['data'][desde:hasta]
    
    # Grafico latido/no latido original y el proceso a traves del modelo
    ax = plt.subplot(5,2,1)
    ax.set_ylim([-2,2])
    plt.plot(this_data)
    plt.grid(True)
    plt.ylabel("input")
    
    out_conv1 = model.conv1(torch.Tensor(this_data).view(1, 1, 24))
    plt.subplot(5,2,2)
    plt.plot(out_conv1[0].tolist()[0], label = "Channel 0")
    plt.plot(out_conv1[0].tolist()[1], label = "Channel 1")
    plt.plot(out_conv1[0].tolist()[2], label = "Channel 2")
    plt.grid(True)
    plt.ylabel("conv1")
    
    out_conv1_act = model.conv1_activation(out_conv1)
    plt.subplot(5,2,3)
    plt.plot(out_conv1_act[0].tolist()[0], label = "Channel 0")
    plt.plot(out_conv1_act[0].tolist()[1], label = "Channel 1")
    plt.plot(out_conv1_act[0].tolist()[2], label = "Channel 2")
    plt.grid(True)
    plt.ylabel("conv1_act")
    
    out_conv2 = model.conv2(out_conv1_act)
    plt.subplot(5,2,4)
    plt.plot(out_conv2[0].tolist()[0], label = "Channel 0")
    plt.plot(out_conv2[0].tolist()[1], label = "Channel 1")
    plt.plot(out_conv2[0].tolist()[2], label = "Channel 2")
    plt.plot(out_conv2[0].tolist()[3], label = "Channel 3")
    plt.plot(out_conv2[0].tolist()[4], label = "Channel 4")
    plt.plot(out_conv2[0].tolist()[5], label = "Channel 5")
    plt.grid(True)
    plt.ylabel("conv2")
    
    out_conv2_act = model.conv2_activation(out_conv2)
    plt.subplot(5,2,5)
    plt.plot(out_conv2_act[0].tolist()[0], label = "Channel 0")
    plt.plot(out_conv2_act[0].tolist()[1], label = "Channel 1")
    plt.plot(out_conv2_act[0].tolist()[2], label = "Channel 2")
    plt.plot(out_conv2_act[0].tolist()[3], label = "Channel 3")
    plt.plot(out_conv2_act[0].tolist()[4], label = "Channel 4")
    plt.plot(out_conv2_act[0].tolist()[5], label = "Channel 5")
    plt.grid(True)
    plt.ylabel("conv2_act")
    
    out_conv3 = model.conv3(out_conv2_act)
    plt.subplot(5,2,6)
    plt.plot(out_conv3[0,0].tolist(), 'x')
    plt.grid(True)
    plt.ylabel("conv3")
    
    out_conv3_act = model.conv3_activation(out_conv3)
    plt.subplot(5,2,7)
    plt.plot(out_conv3_act[0,0].tolist(), 'x')
    plt.grid(True)
    plt.ylabel("conv3_act")
    
    linout_weights = model.linout.weight
    ax = plt.subplot(5,2,8)
    #ax.set_ylim([-1,1])
    plt.plot(linout_weights[0].tolist(), 'x')
    plt.grid(True)
    plt.ylabel("linout_weights")
    
    out_linout = model.linout(out_conv3_act)
    ax = plt.subplot(5,2,9)
    ax.set_ylim([out_linout[0,0,0].item() - 1, out_linout[0,0,0].item() + 1])
    plt.plot([out_linout[0,0,0].item(), out_linout[0,0,0].item()])
    plt.grid(True)
    plt.ylabel("linout_output")
    
    out_model = model.sig(out_linout)
    ax = plt.subplot(5,2,10)
    ax.set_ylim([-0.1, 1.1])
    plt.plot([out_model[0,0,0].item(), out_model[0,0,0].item()])
    plt.grid(True)
    plt.ylabel("model_output")
    
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    
    plt.suptitle("Evolucion del modelo")
    plt.show()
    
    desde = data['no_beat_slices'][5][0]
    hasta = data['no_beat_slices'][5][1]
    this_data = data['data'][desde:hasta]
    
    # Grafico latido/no latido original y el proceso a traves del modelo
    ax = plt.subplot(5,2,1)
    ax.set_ylim([-2,2])
    plt.plot(this_data)
    plt.grid(True)
    plt.ylabel("input")
    
    out_conv1 = model.conv1(torch.Tensor(this_data).view(1, 1, 24))
    plt.subplot(5,2,2)
    plt.plot(out_conv1[0].tolist()[0], label = "Channel 0")
    plt.plot(out_conv1[0].tolist()[1], label = "Channel 1")
    plt.plot(out_conv1[0].tolist()[2], label = "Channel 2")
    plt.grid(True)
    plt.ylabel("conv1")
    
    out_conv1_act = model.conv1_activation(out_conv1)
    plt.subplot(5,2,3)
    plt.plot(out_conv1_act[0].tolist()[0], label = "Channel 0")
    plt.plot(out_conv1_act[0].tolist()[1], label = "Channel 1")
    plt.plot(out_conv1_act[0].tolist()[2], label = "Channel 2")
    plt.grid(True)
    plt.ylabel("conv1_act")
    
    out_conv2 = model.conv2(out_conv1_act)
    plt.subplot(5,2,4)
    plt.plot(out_conv2[0].tolist()[0], label = "Channel 0")
    plt.plot(out_conv2[0].tolist()[1], label = "Channel 1")
    plt.plot(out_conv2[0].tolist()[2], label = "Channel 2")
    plt.plot(out_conv2[0].tolist()[3], label = "Channel 3")
    plt.plot(out_conv2[0].tolist()[4], label = "Channel 4")
    plt.plot(out_conv2[0].tolist()[5], label = "Channel 5")
    plt.grid(True)
    plt.ylabel("conv2")
    
    out_conv2_act = model.conv2_activation(out_conv2)
    plt.subplot(5,2,5)
    plt.plot(out_conv2_act[0].tolist()[0], label = "Channel 0")
    plt.plot(out_conv2_act[0].tolist()[1], label = "Channel 1")
    plt.plot(out_conv2_act[0].tolist()[2], label = "Channel 2")
    plt.plot(out_conv2_act[0].tolist()[3], label = "Channel 3")
    plt.plot(out_conv2_act[0].tolist()[4], label = "Channel 4")
    plt.plot(out_conv2_act[0].tolist()[5], label = "Channel 5")
    plt.grid(True)
    plt.ylabel("conv2_act")
    
    out_conv3 = model.conv3(out_conv2_act)
    plt.subplot(5,2,6)
    plt.plot(out_conv3[0,0].tolist(), 'x')
    plt.grid(True)
    plt.ylabel("conv3")
    
    out_conv3_act = model.conv3_activation(out_conv3)
    plt.subplot(5,2,7)
    plt.plot(out_conv3_act[0,0].tolist(), 'x')
    plt.grid(True)
    plt.ylabel("conv3_act")
    
    linout_weights = model.linout.weight
    ax = plt.subplot(5,2,8)
    #ax.set_ylim([-1,1])
    plt.plot(linout_weights[0].tolist(), 'x')
    plt.grid(True)
    plt.ylabel("linout_weights")
    
    out_linout = model.linout(out_conv3_act)
    ax = plt.subplot(5,2,9)
    ax.set_ylim([out_linout[0,0,0].item() - 1, out_linout[0,0,0].item() + 1])
    plt.plot([out_linout[0,0,0].item(), out_linout[0,0,0].item()])
    plt.grid(True)
    plt.ylabel("linout_output")
    
    out_model = model.sig(out_linout)
    ax = plt.subplot(5,2,10)
    ax.set_ylim([-0.1, 1.1])
    plt.plot([out_model[0,0,0].item(), out_model[0,0,0].item()])
    plt.grid(True)
    plt.ylabel("model_output")
    
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    
    plt.suptitle("Evolucion del modelo")
    plt.show()
    
    """
    while True:
        # Muestro primero un latido y luego un no latido, graficando todas las etapas
        if flag_beat_nobeat:
            # Cargo un latido
            if flag_in_order:
                desde = data['beat_slices'][this_index][0]
                hasta = data['beat_slices'][this_index][1]
            else:
                tup = data['beat_slices'][random.randint(0, len(data['beat_slices']))]
                desde = tup[0]
                hasta = tup[1]
            
            this_data = data['data'][desde:hasta]
            flag_beat_nobeat = False
        else:
            # Cargo un no latido
            if flag_in_order:
                desde = data['no_beat_slices'][this_index][0]
                hasta = data['no_beat_slices'][this_index][1]
                
                this_index += 1
            else:
                tup = data['no_beat_slices'][random.randint(0, len(data['no_beat_slices']))]
                desde = tup[0]
                hasta = tup[1]
            
            this_data = data['data'][desde:hasta]
            
            flag_beat_nobeat = True
        
        # Grafico latido/no latido original y el proceso a traves del modelo
        ax = plt.subplot(5,2,1)
        ax.set_ylim([-2,2])
        plt.plot(this_data)
        plt.grid(True)
        plt.ylabel("input")
        
        out_conv1 = model.conv1(torch.Tensor(this_data).view(1, 1, 24))
        plt.subplot(5,2,2)
        plt.plot(out_conv1[0].tolist()[0], label = "Channel 0")
        plt.plot(out_conv1[0].tolist()[1], label = "Channel 1")
        plt.plot(out_conv1[0].tolist()[2], label = "Channel 2")
        plt.grid(True)
        plt.ylabel("conv1")
        
        out_conv1_act = model.conv1_activation(out_conv1)
        plt.subplot(5,2,3)
        plt.plot(out_conv1_act[0].tolist()[0], label = "Channel 0")
        plt.plot(out_conv1_act[0].tolist()[1], label = "Channel 1")
        plt.plot(out_conv1_act[0].tolist()[2], label = "Channel 2")
        plt.grid(True)
        plt.ylabel("conv1_act")
        
        out_conv2 = model.conv2(out_conv1_act)
        plt.subplot(5,2,4)
        plt.plot(out_conv2[0].tolist()[0], label = "Channel 0")
        plt.plot(out_conv2[0].tolist()[1], label = "Channel 1")
        plt.plot(out_conv2[0].tolist()[2], label = "Channel 2")
        plt.plot(out_conv2[0].tolist()[3], label = "Channel 3")
        plt.plot(out_conv2[0].tolist()[4], label = "Channel 4")
        plt.plot(out_conv2[0].tolist()[5], label = "Channel 5")
        plt.grid(True)
        plt.ylabel("conv2")
        
        out_conv2_act = model.conv2_activation(out_conv2)
        plt.subplot(5,2,5)
        plt.plot(out_conv2_act[0].tolist()[0], label = "Channel 0")
        plt.plot(out_conv2_act[0].tolist()[1], label = "Channel 1")
        plt.plot(out_conv2_act[0].tolist()[2], label = "Channel 2")
        plt.plot(out_conv2_act[0].tolist()[3], label = "Channel 3")
        plt.plot(out_conv2_act[0].tolist()[4], label = "Channel 4")
        plt.plot(out_conv2_act[0].tolist()[5], label = "Channel 5")
        plt.grid(True)
        plt.ylabel("conv2_act")
        
        out_conv3 = model.conv3(out_conv2_act)
        plt.subplot(5,2,6)
        plt.plot(out_conv3[0,0].tolist(), 'x')
        plt.grid(True)
        plt.ylabel("conv3")
        
        out_conv3_act = model.conv3_activation(out_conv3)
        plt.subplot(5,2,7)
        plt.plot(out_conv3_act[0,0].tolist(), 'x')
        plt.grid(True)
        plt.ylabel("conv3_act")
        
        linout_weights = model.linout.weight
        ax = plt.subplot(5,2,8)
        #ax.set_ylim([-1,1])
        plt.plot(linout_weights[0].tolist(), 'x')
        plt.grid(True)
        plt.ylabel("linout_weights")
        
        out_linout = model.linout(out_conv3_act)
        ax = plt.subplot(5,2,9)
        ax.set_ylim([-10, 10])
        plt.plot([out_linout[0,0,0].item(), out_linout[0,0,0].item()])
        plt.grid(True)
        plt.ylabel("linout_output")
        
        out_model = model.sig(out_linout)
        ax = plt.subplot(5,2,10)
        ax.set_ylim([-0.1, 1.1])
        plt.plot([out_model[0,0,0].item(), out_model[0,0,0].item()])
        plt.grid(True)
        plt.ylabel("model_output")
        
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        
        plt.suptitle("Evolucion del modelo")
        plt.show()
        """