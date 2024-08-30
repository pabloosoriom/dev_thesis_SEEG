'''
Este script considera los pasos necesarios para una limpieza apropiada de los datos sEEG. Primero realiza una seleccion de los malos canales, removiéndolos de la 
base de datos, utilizando un filtro a través de un análisis de correlación. Posteriormente se realiza un High-pass filter, y luego un first-order low pass filter 
'''

import numpy as np
import pandas as pd
import mne
from mne import io
from scipy.fft import fft, fftfreq


import numpy as np
import matplotlib.pyplot as plt

def bad_channels_filter(raw, reference_channel, correlation_threshold=0.1):
    # Step 1: Determine bad channels based on correlation
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



def pass_filter(raw, high_pass=1,low_pass=110, verbose=False):
    raw_low_pass = raw.copy().filter(l_freq=high_pass, h_freq=low_pass)
    
    
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

def find_powerline_freqs(raw):
    #Getting the total time of the recording
    total_time = raw.times[-1]
    #Getting the sampling frequency
    sfreq = raw.info['sfreq']
    #Getting the number of channels
    n_channels = raw.info['nchan']

    sampling_rate = sfreq  # Hz
    T = 1.0 / sampling_rate  # Sample spacing
    t = np.arange(0, int(total_time), T)

    miu_s=np.linspace(0.6,0.95,7)

    means=[]

    for miu in miu_s:
        power_line_freqs_pos=[]
        for ch in range(n_channels):
            # Simulating a signal with power line interference at 50 Hz and its harmonics
            signal = raw.get_data()[ch]

            # Perform FFT
            N = len(t)  # Number of samples
            yf = fft(signal)  # Compute the fast Fourier transform
            xf = fftfreq(N, T)[:N // 2]  # Frequencies corresponding to the FFT result

            # Get the magnitude of the FFT (only positive frequencies)
            magnitude = np.abs(yf[:N // 2])
            threshold = np.max(magnitude) * miu
        # Find frequencies where magnitude exceeds the threshold
            power_line_freqs = xf[magnitude > threshold]
            power_line_freqs_pos.append(power_line_freqs[power_line_freqs>0])
        
        means.append(np.mean(np.concatenate(power_line_freqs_pos)))

    print(f"Possible Power line Frequencies Detected: {means}")

    return means


    
def line_noise_filter(raw, notch_freqs, verbose=False):
    raw_notch = raw.copy().notch_filter(freqs=notch_freqs, filter_length='auto', 
                                        notch_widths=None, trans_bandwidth=1, 
                                        method='fir', iir_params=None,
                                        mt_bandwidth=None, p_value=0.05, 
                                        picks=None, n_jobs=5, phase='zero', fir_window='hamming', 
                                        fir_design='firwin', pad='reflect_limited')
    
    if verbose:
        fig, ax = plt.subplots(2)

        raw.plot_psd(ax=ax[0], color='blue',average=True)
        raw_notch.plot_psd(ax=ax[1], color='red',average=True)

        ax[0].set_title('Original signal')
        ax[0].set_ylabel('Power (dB)')
        ax[1].set_title('Signal after power noise removal')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Power (dB)')

        fig.set_tight_layout(True)
        plt.show()

      
    return raw_notch, fig


def tag_high_amplitude(epochs):
    #Threshold for identifying high amplitude transients
    threshold = 3000  # in microvolts (µV)

    # Create a list to store the indices of epochs with high amplitude transients
    bad_epochs = []

    # Iterate over epochs
    for i, epoch in enumerate(epochs):
        # Check if the absolute value of any sample in the epoch exceeds the threshold
        if np.max(np.abs(epoch)) > threshold:
            bad_epochs.append(i)
        

    return bad_epochs


