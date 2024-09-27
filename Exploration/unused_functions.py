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

################################################# For graphs

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
