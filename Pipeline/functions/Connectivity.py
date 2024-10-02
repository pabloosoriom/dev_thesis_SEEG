##Different connectivity measures
import matplotlib.pyplot as plt
import mne
from mne_connectivity import spectral_connectivity_epochs
from sklearn.metrics import pairwise_distances
from nltools.data import Brain_Data, Design_Matrix, Adjacency
import matplotlib.animation as animation
from mne_connectivity import spectral_connectivity_time
import networkx as nx
import seaborn as sns
import numpy as np
from PIL import Image
import json
import io
import gc
from scipy.signal import hilbert

def create_connectivity(epochs, output_path,xyz_loc,method,axises=['r', 'a', 's'],animation=True,state=''):
    ### Important functions###
    def create_frame(matrix, band_name, ch_names, frame_number):
        plt.figure(figsize=(10, 10))
        sns.heatmap(matrix, xticklabels=ch_names, yticklabels=ch_names, cmap='viridis', vmin=0, vmax=1)
        plt.xticks(fontsize=8, rotation=90)
        plt.yticks(fontsize=8)
        plt.title(f'{band_name} - (Epoch) {frame_number}')
        
        # Save the plot to a Pillow image using an in-memory buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    # Create the animation
    def create_animation(array, bands, ch_names,method,details=''):
        for i, band_name in enumerate(bands):
            frames = []
            print(f"Creating animation for band: {band_name}")
            for j in range(array.shape[0]):
                matrix = array[j, :, :, i]
                print(j)
                frame_img = create_frame(matrix, band_name, ch_names, j)
                frames.append(frame_img)

            # Save the frames as an animated GIF
            frames[0].save(output_path+f'{band_name}_{method}{details}_animation.gif', save_all=True, append_images=frames[1:], duration=500, loop=0)

    #Normalize between 0 and 1
    def normalize_matrix(matrix):
        min_value = np.min(matrix)
        max_value = np.max(matrix)
        normalized_matrix = (matrix - min_value) / (max_value - min_value)
        return normalized_matrix
        

   

    if method =='aec&plv':
        #Get the band-pass signal for every frequency of interest
        filtered_epoch_data= frequency_bands(epochs=epochs)
        
        #At first we need to mirror-padd at both ends for applying the Hilbert transform properly
        # Apply to all epochs and channels in a specific band (e.g., 'low_gamma')
        hilbert_transformed_data_bands={}
        envelop_data_bands={}
        phase_data_bands={}

        
        for bands, filtered_epochs in filtered_epoch_data.items():
            n_epochs, n_channels, epoch_len = filtered_epochs.shape
            hilbert_transformed_data = np.empty((n_epochs, n_channels, epoch_len), dtype=np.complex64)
            envelop_list = np.empty((n_epochs, n_channels, epoch_len), dtype=np.float32)
            phase_list = np.empty((n_epochs, n_channels, epoch_len), dtype=np.float32)

            for i, epoch_data in enumerate(filtered_epochs):
                transformed_epoch,envelop,phase = apply_hilbert_transform(epoch_data)
                hilbert_transformed_data[i]=transformed_epoch
                envelop_list[i]=envelop
                phase_list[i]=phase

            hilbert_transformed_data_bands[bands] = hilbert_transformed_data
            envelop_data_bands[bands] = envelop_list
            phase_data_bands[bands] = phase_list

        # Calculate AEC for each frequency band and epoch
        aec_results = {}
        for band, envelopes in envelop_data_bands.items():
            aec_results[band] = []
            print(f'Calculating results for band {band}')
            for envelope in envelopes:                
                aec_matrix = calculate_aec(envelope)
                aec_results[band].append(aec_matrix)
            aec_results[band] = np.array(aec_results[band])

        #Saving a npy file for every band
        for band in aec_results.keys():
            np.save(output_path+f'connectivity_data_{band}_{method[0:3]}_dense.npy', aec_results[band])
        
        del envelop_data_bands
        del hilbert_transformed_data_bands
        del aec_results
        gc.collect()

        
        #Calculate PLV for each frequency band and epoch
        plv_results = {}
        for band, phase in phase_data_bands.items():
            plv_results[band] = []
            print(f'Calculating results for band {band}')
            for phase in phase:                
                plv_matrix = calculate_plv(phase)
                plv_results[band].append(plv_matrix)
            plv_results[band] = np.array(plv_results[band])
        
        
        for band in plv_results.keys():
            np.save(output_path+f'connectivity_data_{band}_{method[4:7]}_dense.npy', plv_results[band])

        del phase_data_bands
        del plv_results
        gc.collect()
    

        

        if animation:
            #Read the data from the npy files
            aec_results = {}
            plv_results = {}
            for band in filtered_epoch_data.keys():
                aec_results[band] = np.load(output_path+f'connectivity_data_{band}_{method[0:3]}_dense.npy')
                plv_results[band] = np.load(output_path+f'connectivity_data_{band}_{method[4:7]}_dense.npy')
            

            aec_results_array = np.array([aec_results[band] for band in aec_results.keys()])
            aec_results_array= np.transpose(aec_results_array,(1,2,3,0))

            plv_results_array = np.array([plv_results[band] for band in plv_results.keys()])
            plv_results_array= np.transpose(plv_results_array,(1,2,3,0))
            print(f'Creating animations for {list(filtered_epoch_data.keys())}')
            create_animation(aec_results_array, list(filtered_epoch_data.keys()), epochs.ch_names,method=method[0:3])
            create_animation(plv_results_array, list(filtered_epoch_data.keys()), epochs.ch_names,method=method[4:7])
    
        #Create correection according to distance 
        # # Calculate the pairwise distances between channels
        #Get the euclidean distance between each pair of electrodes
        distances = pairwise_distances(xyz_loc[axises])
        # Normalize the distances
        normalized_distances = normalize_matrix(distances)
        #Lets multiply the AEC matrix by the normalized distance matrix
        aec_distance = {}
        for band, aec_matrices in aec_results.items():
            aec_distance[band] = []
            for aec_matrix in aec_matrices:
                aec_distance[band].append(aec_matrix * normalized_distances)
            aec_distance[band] = np.array(aec_distance[band])
        
        plv_distance = {}
        for band, plv_matrices in plv_results.items():
            plv_distance[band] = []
            for plv_matrix in plv_matrices:
                plv_distance[band].append(plv_matrix * normalized_distances)
            plv_distance[band] = np.array(plv_distance[band])

        #Save the data
        for band in aec_distance.keys():
            np.save(output_path+f'connectivity_data_{band}_{method[0:3]}_distance_dense.npy', aec_distance[band])
        
        for band in plv_distance.keys():
            np.save(output_path+f'connectivity_data_{band}_{method[4:7]}_distance_dense.npy', plv_distance[band])
        
        #Create animations for the distance corrected data
        if animation:
            print(f'Creating animations for {list(filtered_epoch_data.keys())} with distance correction')
            aec_distance_array = np.array([aec_distance[band] for band in aec_distance.keys()])
            aec_distance_array= np.transpose(aec_distance_array,(1,2,3,0))
            plv_distance_array = np.array([plv_distance[band] for band in plv_distance.keys()])
            plv_distance_array= np.transpose(plv_distance_array,(1,2,3,0))


            create_animation(aec_distance_array, list(filtered_epoch_data.keys()), epochs.ch_names,method=method[0,3],details='_distance_corrected')
            create_animation(plv_distance_array, list(filtered_epoch_data.keys()), epochs.ch_names,method=method[4,7],details='_distance_corrected')
    else: 
        # Freq bands of interest
        # Freq_Bands = {"theta": [4.0, 7.5], "alpha": [7.5, 13.0], 
        #               "beta": [13.0, 30.0],'gamma':[30.0,45.0]}
        Freq_Bands = {"beta+gamma": [13.0, 45.0]}
        n_freq_bands = len(Freq_Bands)
        min_freq = np.min(list(Freq_Bands.values()))
        max_freq = np.max(list(Freq_Bands.values()))

        # Provide the freq points
        freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))

        # The dictionary with frequencies are converted to tuples for the function
        fmin = tuple([list(Freq_Bands.values())[f][0] for f in range(len(Freq_Bands))])
        fmax = tuple([list(Freq_Bands.values())[f][1] for f in range(len(Freq_Bands))])

        sfreq = epochs.info['sfreq']
        # Compute the time-resolved connectivity
        con_time = spectral_connectivity_time(
            epochs, 
            freqs=freqs, 
            method=method, 
            sfreq=sfreq, 
            fmin=fmin,
            fmax=fmax,
            mode="cwt_morlet", 
            faverage=True,
            n_jobs=5
        )
        #Save connectivity data to a file
        np.save(output_path+f'connectivity_data_high_freq_{method}_dense.npy', con_time.get_data(output='dense'))
        con_mat=con_time.get_data(output='dense')
        print(con_mat.shape)



        if animation:
            create_animation(con_mat, Freq_Bands.keys(), epochs.ch_names,method=method)
    return print('Connectivity animation created')

def get_amplitude_envelope(epoch_data):
    analytic_signal = hilbert(epoch_data, axis=1)  # Apply Hilbert transform along time axis
    amplitude_envelope = np.abs(analytic_signal)   # Get amplitude envelope
    return amplitude_envelope


def frequency_bands(epochs):
    frequency_bands = {
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (15, 25),
    'low_gamma': (35, 50),
    'high_gamma1': (70, 110)
    }

    filtered_epochs = {}
    for band, (l_freq, h_freq) in frequency_bands.items():
        filtered_epochs[band] = epochs.copy().filter(l_freq, h_freq, fir_design='firwin', phase='zero-double',n_jobs=5)
        #No more frequencies are considered to accomplish the Nysquist-shannon sampling theorem
    
    return filtered_epochs

def get_envelop(epoch_data):
    return np.abs(epoch_data)



# Function for mirror padding and Hilbert transform
def apply_hilbert_transform(epoch_data):
    # Mirror padding the epoch data (pad by mirroring the first and last samples)
    padding_length = 1000  # Padding with half of the epoch length
    padded_data = np.pad(epoch_data, ((0, 0), (padding_length, padding_length)), mode='reflect')

    # Apply Hilbert transform
    analytic_signal = hilbert(padded_data, axis=1)

    # Trim the padding to retrieve the original epoch length
    trimmed_analytic_signal = analytic_signal[:, padding_length:-padding_length]
    envelop=np.abs(trimmed_analytic_signal)
    phase=np.angle(trimmed_analytic_signal)

    return trimmed_analytic_signal,envelop,phase



def calculate_aec(amplitude_envelopes):
    n_channels = amplitude_envelopes.shape[0]
    aec_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            correlation = np.corrcoef(amplitude_envelopes[i], amplitude_envelopes[j])[0, 1]
            aec_matrix[i, j] = correlation
            aec_matrix[j, i] = correlation  # Symmetric matrix
            
    return aec_matrix

def calculate_plv(phase_data):
    n_channels = phase_data.shape[0]
    plv_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            phase_diff = phase_data[i]-phase_data[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv  # Symmetric matrix
            
    return plv_matrix


def create_granger_regions(epochs,freqs,fmin,fmax,sfreq):
    # Group channels based on their prefixes
    channel_groups = {}
    for idx, channel in enumerate(epochs.ch_names):
        prefix = ''.join(filter(str.isalpha, channel))
        if prefix in channel_groups:
            channel_groups[prefix].append(idx)
        else:
            channel_groups[prefix] = [idx]


    for prefix, channels in channel_groups.items():
        print(f'{prefix}: {channels}')

    
    def calculate_and_plot_granger_causality(epochs, signals_a, signals_b, freqs, fmin, fmax, sfreq):
        indices_ab = (np.array([signals_a]), np.array([signals_b]))  # A => B
        indices_ba = (np.array([signals_b]), np.array([signals_a]))  # B => A


        gc_ab = spectral_connectivity_time(
            epochs, 
            freqs=freqs, 
            indices=indices_ab,
            method='gc', 
            sfreq=sfreq, 
            fmin=fmin,
            fmax=fmax,
            mode="cwt_morlet", 
            faverage=True,
            n_jobs=5
            
        )  # A => B

        gc_ba = spectral_connectivity_time(
            epochs, 
            freqs=freqs, 
            indices=indices_ba,
            method='gc', 
            sfreq=sfreq, 
            fmin=fmin,
            fmax=fmax,
            mode="cwt_morlet", 
            faverage=True,
            n_jobs=5
            
        )# B => A
        return gc_ab, gc_ba
    #Create a dictionary to store the Granger causality values for every pair of prefixes
    gc_values = {}
    for j, prefix1 in enumerate(channel_groups.keys()):
        for k, prefix2 in enumerate(channel_groups.keys()):
            if j != k:
               print(f"Calculating GC for {prefix1} => {prefix2}")
               A=channel_groups[prefix1]
               B=channel_groups[prefix2]
               gc_ab, gc_ba = calculate_and_plot_granger_causality(epochs, A, B, freqs, fmin, fmax, sfreq)
               gc_values[(prefix1,prefix2)] = gc_ab.get_data(output='dense')
               gc_values[(prefix2,prefix1)] = gc_ba.get_data(output='dense')
    
    return gc_values


    

