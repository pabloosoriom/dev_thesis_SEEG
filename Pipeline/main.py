import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from nltools.data import Brain_Data, Design_Matrix, Adjacency
import networkx as nx
from scipy import signal
from mne_connectivity import spectral_connectivity_epochs
import seaborn as sns

from functions.EpiIndex import *
from functions.Connectivity import *
from functions.preprocessing import *


import sys
import time

mne.set_config('MNE_MEMMAP_MIN_SIZE', '10M') 
mne.set_config('MNE_CACHE_DIR', '/dev/shm')

def main():
    # Start measuring execution time
    start_time = time.time()

    # Redirect output to a file
    patient = 'pte_01'
    output_path = '/home/pablo/works/dev_thesis_SEEG//outputs/'+patient+'/'
    input_path = '/home/pablo/works/dev_thesis_SEEG/data/'+patient+'/'+'sets/segments_ictal_SR/'

    

    # # #Reading the data
    raw = mne.io.read_raw_fif(input_path + 'ictal-epo_6' + '.fif', preload=True)

     #Additional steps 
    #Eliminate a channel
    # raw_cleaned.drop_channels(['EKG+'])

    #Pass filter 
    raw_filtered1,_=pass_filter(raw)

    # #Find the Possible Power Line Frequencies
    # possible_freqs=find_powerline_freqs(raw)
    # plf=np.mean(possible_freqs)

    #Remove Possible Power Line Frequencies 
    raw_filtered,_ = line_noise_filter(raw_filtered1,[60,120],True)    
     
    #Setting names
    raw_filtered = set_names(raw_filtered)

    #Save filtered raw
    raw_filtered.save(output_path + patient + '_filtered.fif', overwrite=True)

   

    # # # Epoching
    raw = mne.io.read_raw_fif(output_path + patient + '_filtered.fif', preload=True)
    t_sec=raw.n_times/raw.info['sfreq']
    epochs=mne.make_fixed_length_epochs(raw, duration=6, preload=True)


    print(epochs.info)   

    #Tagging epochs with high amplitude transients
    bad_epochs=tag_high_amplitude(epochs)

    if bad_epochs:
        epochs.drop(bad_epochs)

    
    #method
    method = 'aec'
    print(f'Connectivity method: {method}')
    # # # Connectivity
    # Define frequency bands
    # Create the connectivity animation
    create_connectivity(
        epochs=epochs,
        output_path=output_path + patient + '_',
        method=method,
        animation=False
    )

    # # print('Problematic channels dropped from the main database')

    # # channels = raw.ch_names

    # # Ei_n1, ER_matrix1, U_n_matrix, ER_n_array, alarm_time = get_EI(raw)

    # # plotting_ei(Ei_n1, ER_matrix1, channels, save_path='.lightning_studio/EpiPlan /Pipeline/outputs')

    # # # Save Ei_n1 matrix to CSV
    # # np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/Ei_n1.csv', Ei_n1, delimiter=';')

    # # # Save ER_matrix1 matrix to CSV
    # # np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/ER_matrix1.csv', ER_matrix1, delimiter=';')

    # # # Save U_n_matrix matrix to CSV
    # # np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/U_n_matrix.csv', U_n_matrix, delimiter=';')

    # # # Save ER_n_array matrix to CSV
    # # np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/ER_n_array.csv', ER_n_array, delimiter=';')

    # # # Save alarm_time matrix to CSV
    # # np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/alarm_time.csv', alarm_time, delimiter=';')
    # # # Stop measuring execution time
    end_time = time.time()
    execution_time = end_time - start_time

    print(f'Total execution time: {execution_time} seconds')

    # # # Print and save the execution time
    # # print(f'Total execution time: {execution_time} seconds')
    # # with open('.lightning_studio/EpiPlan/Pipeline/output.txt', 'a') as f:
    # #     f.write(f'\nTotal execution time: {execution_time} seconds\n')

    # # Close the redirected output file
    # # sys.stdout.close()

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



def plotting_ei(Ei_n, ER_matrix, channels, derivatives_d1=None, save_path=None):
    if derivatives_d1 is None:
        plt.imshow(ER_matrix, cmap='viridis', interpolation='bicubic', aspect='auto', extent=[0, 40000, 0, 22])
        plt.colorbar()
        plt.yticks(np.arange(len(channels)), channels)
        plt.xlabel('Window number')
        plt.ylabel('Channel name')
        plt.title('ER_n')
        if save_path:
            plt.savefig(save_path + '_ER_matrix.png')  # Save ER_matrix plot
        else:
            plt.show()

        plt.figure(figsize=(20, 5))
        plt.bar(channels, Ei_n)
        plt.xlabel('Channel name')
        plt.xticks(rotation=90)
        plt.ylabel('EI')
        if save_path:
            plt.savefig(save_path + '_barplot.png')  # Save barplot
        else:
            plt.show()
    else:
        plt.imshow(ER_matrix, cmap='viridis', interpolation='bicubic', aspect='auto', extent=[0, 40000, 0, 22])
        plt.colorbar()
        plt.yticks(np.arange(len(channels)), channels)
        plt.xlabel('Window number')
        plt.ylabel('Channel name')
        plt.title('ER_n')
        if save_path:
            plt.savefig(save_path + '_ER_matrix.png')  # Save ER_matrix plot
        else:
            plt.show()

        plt.bar(channels, Ei_n)
        plt.xlabel('Channel name')
        plt.ylabel('EI')
        if save_path:
            plt.savefig(save_path + '_barplot.png')  # Save barplot
        else:
            plt.show()

        plt.imshow(derivatives_d1, cmap='viridis', interpolation='bicubic', aspect='auto', extent=[0, 40000, 0, 22])
        plt.colorbar()
        plt.yticks(np.arange(len(channels)), channels)
        plt.xlabel('Window number')
        plt.ylabel('Channel name')
        plt.title('ER_n')
        if save_path:
            plt.savefig(save_path + '_derivatives.png')  # Save derivatives plot
        else:
            plt.show()

if __name__ == "__main__":
    main()
# segment_processing()