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


    ######################Reading data###################
    # Start measuring execution time
    start_time = time.time()

    # Redirect output to a file
    patient = 'pte_01'
    output_path = '/home/pablo/works/dev_thesis_SEEG//outputs/'+patient+'/'
    input_path = '/home/pablo/works/dev_thesis_SEEG/data/'+patient+'/'+'sets/segments_ictal_SR/'

    
    
    # # #Reading the signal data
    raw = mne.io.read_raw_fif(input_path + 'ictal-epo_6' + '.fif', preload=True)
    # #Reading xyz schema
    xyz_loc = pd.read_csv('/home/pablo/works/dev_thesis_SEEG/data/'+patient+'/others/'+'sEEG_locs.tsv',sep='\t')

    # # # Formating both files to have according channels 

    raw , xyz_loc = format_data(raw, xyz_loc)

    plot_xyz(xyz_loc,outputpath=output_path)


    ######################Preprocessing###################
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

    

    ######################Connectivity###################
    raw = mne.io.read_raw_fif(output_path + patient + '_filtered.fif', preload=True)
    t_sec=raw.n_times/raw.info['sfreq']
    epochs=mne.make_fixed_length_epochs(raw, duration=30, preload=True)
 

    #Tagging epochs with high amplitude transients
    bad_epochs=tag_high_amplitude(epochs)

    if bad_epochs:
        epochs.drop(bad_epochs)


    method = 'aec&plv'
    print(f'Connectivity method: {method}')
    # # # Connectivity
    # Define frequency bands
    # Create the connectivity animation
    create_connectivity(
        epochs=epochs,
        output_path=output_path + patient + '_',
        method=method,
        xyz_loc=xyz_loc,
        animation=False
    )

    ###################Community Detection###################
    







    
    end_time = time.time()
    execution_time = end_time - start_time

    print(f'Total execution time: {execution_time} seconds')




if __name__ == "__main__":
    main()
# segment_processing()