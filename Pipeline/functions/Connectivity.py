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
from scipy.signal import hilbert



def compute_connectivity(raw, window_duration=100, overlap=0.25, plot=True):
    
    epochs = mne.make_fixed_length_epochs(raw, duration=window_duration, overlap=overlap, preload=True)
    times = epochs.times
    ch_names = epochs.ch_names

    fmin, fmax = 4., 9.  # compute connectivity within 4-9 Hz
    sfreq = raw.info['sfreq']  # sampling frequency
    tmin = 0.0  # exclude the baseline period

    # Compute PLI, wPLI, and dPLI
    con_pli = spectral_connectivity_epochs(
        epochs, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin,
        fmax=fmax, faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

    con_wpli = spectral_connectivity_epochs(
        epochs, method='wpli', mode='multitaper', sfreq=sfreq, fmin=fmin,
        fmax=fmax, faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

    con_dpli = spectral_connectivity_epochs(
        epochs, method='dpli', mode='multitaper', sfreq=sfreq, fmin=fmin,
        fmax=fmax, faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        axs[0].imshow(con_pli.get_data('dense'), vmin=0, vmax=1)
        axs[0].set_title("PLI")
        axs[0].set_ylabel("Node 1")
        axs[0].set_xlabel("Node 2")

        axs[1].imshow(con_wpli.get_data('dense'), vmin=0, vmax=1)
        axs[1].set_title("wPLI")
        axs[1].set_xlabel("Node 2")

        im = axs[2].imshow(con_dpli.get_data('dense'), vmin=0, vmax=1)
        axs[2].set_title("dPLI")
        axs[2].set_xlabel("Node 2")

        fig.colorbar(im, ax=axs.ravel())
        plt.show()

    return con_pli.get_data('dense'), con_wpli.get_data('dense'), con_dpli.get_data('dense')



def compute_coherence(raw, fmin=(8., 13.), fmax=(13., 30.), plot=True):
    sfreq = raw.info['sfreq']  # sampling frequency
    tmin = 0.0  # exclude the baseline period
    
    epochs = mne.make_fixed_length_epochs(raw, duration=100, overlap=0.25, preload=True)
    
    coh = spectral_connectivity_epochs(
        epochs, method='coh', mode='fourier', sfreq=sfreq, fmin=fmin,
        fmax=fmax, faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)
    
    freqs = coh.freqs
    
    print('Frequencies in Hz over which coherence was averaged for alpha:')
    print(freqs[0])
    print('Frequencies in Hz over which coherence was averaged for beta:')
    print(freqs[1])
    
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        axs[0].imshow(coh.get_data('dense')[:,:,0], vmin=0, vmax=1)
        axs[0].set_title("Alpha")
        axs[0].set_ylabel("Node 1")
        axs[0].set_xlabel("Node 2")

        im = axs[1].imshow(coh.get_data('dense')[:,:,1], vmin=0, vmax=1)
        axs[1].set_title("Beta")
        axs[1].set_xlabel("Node 2")

        fig.colorbar(im, ax=axs.ravel())
        plt.show()
    
    return coh.get_data('dense')


def compute_adjacency_graph(raw,data=None, matrix_name=None, threshold=0.5, plot=True):
    chanels = raw.ch_names
    if data is None:
        # Use correlation matrix if no specific matrix is provided
        matrix_name = "Correlation"
        # Assuming you have 'raw' available
        data = []
        for i in range(0, len(chanels)):
            data.append(raw.get_data(picks=chanels[i]))

        #Pearson correlation
        corr=[]
        for i in range(0, len(chanels)):
            for j in range(0, len(chanels)):
                corr.append(np.corrcoef(data[i][0], data[j][0])[0][1])

        #Plotting the correlation matrix
        corr=np.array(corr)
        corr=corr.reshape(len(chanels), len(chanels))
        matrix = corr
    else:
        if matrix_name is None:
            raise ValueError("Please provide the name of the matrix.")
        matrix = data
    
    # if plot:
    #     sns.heatmap(matrix, square=True, vmin=-1, vmax=1, cmap='RdBu_r')
    #     plt.title(f'{matrix_name} Matrix')
    #     plt.show()
    
    a = Adjacency(matrix,labels=[x for x in chanels])
    print(f'Adjacency matrix for {matrix_name} matrix')
    a_thresholded = a.threshold(upper=threshold, binarize=True)
    
    if plot:
        G=plot_adjacency_graph(a_thresholded)
    
    return a_thresholded,G

def plot_adjacency_graph(adjacency):
    plt.figure(figsize=(20, 10))
    G = adjacency.to_graph()
    pos = nx.kamada_kawai_layout(G)
    node_and_degree = G.degree()
    
    nx.draw_networkx_edges(G, pos, width=3, alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=14, font_color='darkslategray')
    
    nx.draw_networkx_nodes(G, pos, nodelist=list(dict(node_and_degree).keys()),
                           node_size=[x[1]*100 for x in node_and_degree],
                           node_color=list(dict(node_and_degree).values()),
                           cmap=plt.cm.Reds_r, linewidths=2, edgecolors='darkslategray', alpha=1)
    plt.title('Adjacency Graph')
    plt.show()

    plt.hist(dict(G.degree).values(), bins=20, color='lightseagreen', alpha=0.7)
    plt.ylabel('Frequency', fontsize=18)
    plt.xlabel('Degree', fontsize=18)

    plt.show()

    #Degree per channel plot
    plt.figure(figsize=(20,15))
    plt.barh(list(dict(G.degree).keys()), list(dict(G.degree).values()), color='lightseagreen')
    plt.xlabel('Degree', fontsize=18)
    plt.ylabel('Channel', fontsize=18)
    plt.title('Degree per channel', fontsize=20)
    plt.show()
    return G



# def compute_adjacency_graph(raw, data=None, matrix_name=None, plot=True):
#     channels = raw.ch_names
#     if data is None:
#         # Use correlation matrix if no specific matrix is provided
#         matrix_name = "Correlation"
#         data = []
#         for i in range(len(channels)):
#             data.append(raw.get_data(picks=channels[i]))

#         # Pearson correlation
#         corr = []
#         for i in range(len(channels)):
#             for j in range(len(channels)):
#                 corr.append(np.corrcoef(data[i][0], data[j][0])[0][1])

#         # Reshape the correlation list to a matrix
#         corr = np.array(corr).reshape(len(channels), len(channels))
#         matrix = corr
#     else:
#         if matrix_name is None:
#             raise ValueError("Please provide the name of the matrix.")
#         matrix = data
    
#     if plot:
#         sns.heatmap(matrix, square=True, vmin=-1, vmax=1, cmap='RdBu_r')
#         plt.title(f'{matrix_name} Matrix')
#         plt.show()
    
#     a = Adjacency(matrix, labels=[x for x in channels])
#     print(f'Adjacency matrix for {matrix_name} matrix')

#     if plot:
#         G = plot_adjacency_graph(a)
    
#     return a, G

# def plot_adjacency_graph(adjacency):
#     plt.figure(figsize=(20, 10))
#     G = adjacency.to_graph()
#     pos = nx.kamada_kawai_layout(G)
#     node_and_degree = G.degree()
    
#     # Draw edges with weights
#     edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
#     nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w*5 for w in weights], alpha=0.5)
    
#     # Draw nodes and labels
#     nx.draw_networkx_labels(G, pos, font_size=14, font_color='darkslategray')
#     nx.draw_networkx_nodes(G, pos, nodelist=list(dict(node_and_degree).keys()),
#                            node_size=[x[1]*100 for x in node_and_degree],
#                            node_color=list(dict(node_and_degree).values()),
#                            cmap=plt.cm.Reds_r, linewidths=2, edgecolors='darkslategray', alpha=1)
#     plt.title('Adjacency Graph')
#     plt.show()

#     # Degree histogram
#     plt.hist(dict(G.degree).values(), bins=20, color='lightseagreen', alpha=0.7)
#     plt.ylabel('Frequency', fontsize=18)
#     plt.xlabel('Degree', fontsize=18)
#     plt.show()

#     # Degree per channel plot
#     plt.figure(figsize=(20, 15))
#     plt.barh(list(dict(G.degree).keys()), list(dict(G.degree).values()), color='lightseagreen')
#     plt.xlabel('Degree', fontsize=18)
#     plt.ylabel('Channel', fontsize=18)
#     plt.title('Degree per channel', fontsize=20)
#     plt.show()

#     return G


def calculate_and_plot_granger_causality(epochs, signals_a, signals_b,verbose=True, fmin=5, fmax=30, gc_n_lags=20,plot=True):
    indices_ab = (np.array([signals_a]), np.array([signals_b]))  # A => B
    indices_ba = (np.array([signals_b]), np.array([signals_a]))  # B => A


    gc_ab = spectral_connectivity_epochs(
        epochs,
        method=["gc"],
        indices=indices_ab,
        fmin=fmin,
        fmax=fmax,
        rank=(np.array([5]), np.array([5])),
        gc_n_lags=gc_n_lags,
        verbose=verbose,
    )  # A => B

    gc_ba = spectral_connectivity_epochs(
        epochs,
        method=["gc"],
        indices=indices_ba,
        fmin=fmin,
        fmax=fmax,
        rank=(np.array([5]), np.array([5])),
        gc_n_lags=gc_n_lags,
        verbose=verbose,
        )  # B => A

    freqs = gc_ab.freqs

    # Plot GC: [A => B]
    if plot == True:
        fig, axis = plt.subplots(1, 1)
        axis.plot(freqs, gc_ab.get_data()[0], linewidth=2, label='A => B')
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Connectivity (A.U.)")
        fig.suptitle("GC: [A => B] and [B => A]")

        # Plot GC: [B => A]
        axis.plot(freqs, gc_ba.get_data()[0], linewidth=2, label='B => A')
        axis.legend()
        plt.show()

        # Plot Net GC: [A => B] - [B => A]
        net_gc = gc_ab.get_data() - gc_ba.get_data()  # [A => B] - [B => A]
        fig, axis = plt.subplots(1, 1)
        axis.plot((freqs[0], freqs[-1]), (0, 0), linewidth=2, linestyle="--", color="k")
        axis.plot(freqs, net_gc[0], linewidth=2)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Connectivity (A.U.)")
        fig.suptitle("Net GC: [A => B] - [B => A]")
        plt.show()

    return gc_ab, gc_ba, freqs



   


def create_connectivity(epochs, output_path,method='pli',animation=False,state=''):
    ### Important functions###
    def create_frame(matrix, band_name, ch_names, frame_number):
        plt.figure(figsize=(10, 10))
        sns.heatmap(matrix, xticklabels=ch_names, yticklabels=ch_names, cmap='viridis')
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
    def create_animation(array, bands, ch_names,method):
        for i, band_name in enumerate(bands):
            frames = []
            print(f"Creating animation for band: {band_name}")
            for j in range(array.shape[0]):
                matrix = array[j, :, :, i]
                print(j)
                frame_img = create_frame(matrix, band_name, ch_names, j)
                frames.append(frame_img)

            # Save the frames as an animated GIF
            frames[0].save(output_path+f'{band_name}_{method}_animation.gif', save_all=True, append_images=frames[1:], duration=500, loop=0)
    



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

    if method == 'pli' :
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

    elif method =='aec':
        #Get the band-pass signal for every frequency of interest
        filtered_epochs= frequency_bands(epochs=epochs)
        
        #At first we need to mirror-padd at both ends for applying the Hilbert transform properly
        # Apply to all epochs and channels in a specific band (e.g., 'low_gamma')
        hilbert_transformed_data_bands={}
        envelop_data_bands={}
        phase_data_bands={}
        for bands in filtered_epochs.keys():
            hilbert_transformed_data = []
            envelop_list=[]
            phase_list=[]
            for epoch_data in filtered_epochs[bands]:
                transformed_epoch,envelop,phase = apply_hilbert_transform(epoch_data)
                hilbert_transformed_data.append(transformed_epoch)
                envelop_list.append(envelop)
                phase_list.append(phase)

            hilbert_transformed_data = np.array(hilbert_transformed_data)  # Shape: (n_epochs, n_channels, epoch_length)
            hilbert_transformed_data_bands[bands]=hilbert_transformed_data
            envelop_list=np.array(envelop_list)
            envelop_data_bands[bands]=envelop_list
            phase_list=np.array(phase_list)
            phase_data_bands[bands]=phase_list

        # Calculate AEC for each frequency band and epoch
        aec_results = {}
        for band, envelopes in envelop_data_bands.items():
            aec_results[band] = []
            print(f'Calculating results for band {band}')
            for envelope in envelopes:                
                aec_matrix = calculate_aec(envelope)
                aec_results[band].append(aec_matrix)
            aec_results[band] = np.array(aec_results[band])
        #Transform the dictionary in a four dimenstional matrix, where the 4th dimension is the band
        #Saving a npy file for every band
        for band in aec_results.keys():
            np.save(output_path+f'connectivity_data_{band}_{method}_dense.npy', aec_results[band])

        aec_results = np.array([aec_results[band] for band in aec_results.keys()])
        aec_results= np.transpose(aec_results,(1,2,3,0))
        # # # Create a new MNE Epochs object with the trimmed data
        # # info = epochs.info  # Keep the original info
        # # new_epochs = mne.EpochsArray(trimmed_data, info, epoch_time=epochs.times)
        print(f'Creating animations for {list(filtered_epochs.keys())}')
        create_animation(aec_results, list(filtered_epochs.keys()), epochs.ch_names,method=method)
    elif method == 'gc':

        print('Granger causality')
        dict_gc = create_granger_regions(epochs,freqs,fmin,fmax,sfreq)
        #Save dictionary
        np.save(output_path+f'connectivity_data_low_freq_{method}_dense.npy', dict_gc)
        with open(output_path+f'connectivity_data_low_freq_{method}_dense.json', 'w') as json_file:
            json.dump(dict_gc, json_file)
     # Generate all possible pairs of channels (excluding self-connections)
        # channels = epochs.ch_names
        # n_channels = len(channels)
        # n_epochs = len(epochs)

        # # Initialize a 3D matrix to store Granger causality values
        # gc_matrix = np.zeros((n_epochs, n_channels, n_channels))

        # # Loop over each epoch
        # for epoch_idx in range(n_epochs):
        #     print(f"Processing epoch {epoch_idx + 1}/{n_epochs}...")
            
        #     # Generate all possible pairs of channels (excluding self-connections)
        #     indices = [(np.array(i), np.array(j)) for i in range(n_channels) for j in range(n_channels) if i != j]
        #     print(indices)
            
        #     # Extract the data for the current epoch
        #     current_epoch = epochs[epoch_idx]
            
        #     # Calculate Granger causality for each pair of channels
        #     for set_idx in indices:
        #         con = spectral_connectivity_epochs(
        #             current_epoch,
        #             method="gc",
        #             indices=set_idx,
        #             fmin=fmin,
        #             fmax=fmax,
        #             mode="cwt_morlet", 
        #             faverage=True,
        #             n_jobs=5
        #         )
        #         # Extract Granger causality value
        #         gc_value = con.get_data(output="dense")[0, 0]
                
        #         # Store the value in the matrix
        #         i, j = set_idx[0][0], set_idx[1][0]
        #         gc_matrix[epoch_idx, i, j] = gc_value
    # # Create animation for every band in the bands dictionary
    if animation:
        create_animation(con_mat, Freq_Bands.keys(), epochs.ch_names,method=method)
    else:
        print('Connectivity data saved')
        # #Create a figure with averaged data over all epochs
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # matrix=con_mat[:,:,:,0]
        # sns.heatmap(np.mean(matrix, axis=0), xticklabels=epochs.ch_names, yticklabels=epochs.ch_names, cmap='viridis')
        # plt.xticks(fontsize=8, rotation=90)
        # plt.yticks(fontsize=8)
        # plt.title(f'{method} connectivity for all epochs in {state} state ')
        # #save the figure
        # plt.savefig(output_path+f'{method}_connectivity.png')
        
    # return con_time
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


    

