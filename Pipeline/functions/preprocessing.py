'''
Este script considera los pasos necesarios para una limpieza apropiada de los datos sEEG. Primero realiza una seleccion de los malos canales, removiéndolos de la 
base de datos, utilizando un filtro a través de un análisis de correlación. Posteriormente se realiza un High-pass filter, y luego un first-order low pass filter 
'''

import numpy as np
import pandas as pd
import mne
from mne import io


import numpy as np
import matplotlib.pyplot as plt

def clean_data(raw, reference_channel, correlation_threshold=0.1):
    # Step 1: Determine outlier channels based on correlation
    ch_names = raw.ch_names
    correlation_values = np.corrcoef(raw[reference_channel][0], raw.get_data())[0, 1:]
    outlier_channels = np.where(np.abs(correlation_values) < correlation_threshold)[0]
    
    # Step 2: Plot correlation values to visualize outliers
    fig, ax = plt.subplots()
    ax.plot(correlation_values, 'ro')
    ax.axhline(correlation_threshold, color='k')
    ax.axhline(-correlation_threshold, color='k')
    ax.set(title='Outlier detection', xlabel='Channel index', ylabel='Correlation coefficient')
    ax.plot(outlier_channels, correlation_values[outlier_channels], 'r.', markersize=12)
    plt.show()
    
    # Step 3: Identify problematic channels
    problematic_channels = [ch_names[idx] for idx in outlier_channels]
    print(f'Problematic channels: {", ".join(problematic_channels)}')
    
    # Step 4: Plot PSD of outlier channels
    raw.plot_psd(picks=outlier_channels, average=False)
    
    # Step 5: Drop problematic channels and describe the cleaned data
    raw_cleaned = raw.copy().drop_channels(problematic_channels)
    raw_cleaned.describe()
    
    # Step 6: Plot comparison of data before and after cleaning
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1 = axes[0]
    ax1.set_title('Before Cleaning')
    raw.plot_psd(ax=ax1)
    
    ax2 = axes[1]
    ax2.set_title('After Cleaning')
    raw_cleaned.plot_psd(ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    return raw_cleaned, fig

def high_pass_filter(raw, high_pass_freq = 0.16 , verbose=True):
    raw_high_pass = raw.copy().filter(l_freq=high_pass_freq, h_freq=None)
    
    if verbose:
        fig = raw.plot_psd()
        fig = raw_high_pass.plot_psd()
    
    return raw_high_pass, fig

def low_pass_filter(raw, high_cutoff=97, verbose=True):
    raw_low_pass = raw.copy().filter(l_freq=None, h_freq=high_cutoff)
    
    if verbose:
        fig = raw.plot_psd()
        fig = raw_low_pass.plot_psd()
    
    return raw_low_pass, fig

def set_names(raw, type='seeg'):
    # Get the channel names
    ch_names = raw.ch_names 
    # Dictionary to hold channel types
    channel_types = {}

    # Set all channel types to 'seeg'
    for ch_name in ch_names:
        channel_types[ch_name] = 'seeg'

    # Set the channel types
    raw.set_channel_types(channel_types)
    print('Channel types set to "seeg"')
    return raw


# # #Testing filter functions
# raw_cleaned = mne.io.read_raw_fif('/home/pablo/works/dev_thesis_SEEG/data/pte_6_cleaned.fif', preload=True)
# # #raw_cleaned, fig = clean_data(raw, 'Cz')
# raw_high_pass, fig1 = high_pass_filter(raw_cleaned)
# raw_low_pass, fig2 = low_pass_filter(raw_high_pass)
# print(raw_low_pass)
# print(raw_low_pass.info)
# # #Save the figures 
# fig1.savefig('/home/pablo/works/dev_thesis_SEEG/data/pte_6_high_pass_filter.png')
# fig2.savefig('/home/pablo/works/dev_thesis_SEEG/data/pte_6_low_pass_filter.png')
# raw_low_pass = set_names(raw_low_pass)
# raw_low_pass.save('/home/pablo/works/dev_thesis_SEEG/data/pte_6_filtered.fif', overwrite=True)