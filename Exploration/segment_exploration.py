import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from nltools.data import Brain_Data, Design_Matrix, Adjacency
import networkx as nx
from scipy import signal
from mne_connectivity import spectral_connectivity_epochs
import seaborn as sns


def segment_processing():
     # Start measuring execution time
    start_time = time.time()
    # Redirect output to a file
    output_path = '/home/pablo/works/dev_thesis_SEEG/data/outputs/segments/'
    input_path = '/home/pablo/works/dev_thesis_SEEG/data/segments/'
    

    # # #Filtering
    interictal_no_spikes=mne.io.read_raw_fif(input_path + 'interictal_no_spikes-epo_1' + '.fif', preload=True)
    interictal_spikes=mne.io.read_raw_fif(input_path + 'interictal_spikes-epo_1' + '.fif', preload=True)
    preictal=mne.io.read_raw_fif(input_path + 'preictal-epo_1' + '.fif', preload=True)
    ictal=mne.io.read_raw_fif(input_path + 'ictal-epo_1' + '.fif', preload=True)
    postictal=mne.io.read_raw_fif(input_path + 'postictal-epo_1' + '.fif', preload=True)
    
    data=[interictal_no_spikes, interictal_spikes, preictal, ictal, postictal]
    names=['interictal_no_spikes', 'interictal_spikes', 'preictal', 'ictal', 'postictal']
    
    filtered_data = []
    #Filters for each each segment
    for i in range(5):
        raw_cleaned = data[i]
        raw_high_pass, fig1 = high_pass_filter(raw_cleaned)
        raw_low_pass, fig2 = low_pass_filter(raw_high_pass)
        raw_low_pass = set_names(raw_low_pass)
        raw_low_pass.save(output_path + names[i] + '_filtered.fif', overwrite=True)
        filtered_data.append(raw_low_pass)

    print('Data filtered')

    
    # # # Epoching
    epochs_data = []
    for i in range(5):
        raw = filtered_data[i]
        t_sec=raw.n_times/raw.info['sfreq']
        epochs=mne.make_fixed_length_epochs(raw, duration=t_sec/20, preload=True)
        epochs_data.append(epochs)
        print('Epoching segment ' + names[i] + ' done')
    
    print('Data segmented')
    
    #method
    method = 'coh'
    print(f'Connectivity method: {method}')
    # # # Connectivity
    #Iterate over the segments
    for i in range(5):
        # Create the connectivity animation
        create_connectivity(
            epochs=epochs_data[i],
            output_path=output_path + names[i] + '_',
            method=method,
            animation=False,
            state=names[i]
        )
        print('Connectivity segment ' + names[i] + ' done')
    
    end_time = time.time()
    execution_time = end_time - start_time

    print(f'Total execution time: {execution_time} seconds')
