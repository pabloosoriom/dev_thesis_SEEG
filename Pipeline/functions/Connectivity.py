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
import io



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
    
    if plot:
        sns.heatmap(matrix, square=True, vmin=-1, vmax=1, cmap='RdBu_r')
        plt.title(f'{matrix_name} Matrix')
        plt.show()
    
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

def create_connectivity_animation(epochs, output_path,method='pli'):
   # Freq bands of interest
    # Freq_Bands = {"theta": [4.0, 7.5], "alpha": [7.5, 13.0], 
    #               "beta": [13.0, 30.0],'gamma':[30.0,45.0]}
    Freq_Bands = {"alpha": [7.5, 45]}
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
        n_jobs=5,
        n_cycles=1   
    )

    #Save connectivity data to a file
    np.save(output_path+'connectivity_data_dense.npy', con_time.get_data(output='dense'))
    con_mat=con_time.get_data(output='dense')
    print(con_mat.shape)
        
    # Create animation for every band in the bands dictionary
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
    def create_animation(array, bands, ch_names):
        for i, band_name in enumerate(bands):
            frames = []
            print(f"Creating animation for band: {band_name}")
            for j in range(array.shape[0]):
                matrix = array[j, :, :, i]
                print(j)
                frame_img = create_frame(matrix, band_name, ch_names, j)
                frames.append(frame_img)

            # Save the frames as an animated GIF
            frames[0].save(output_path+f'{band_name}_animation.gif', save_all=True, append_images=frames[1:], duration=500, loop=0)
    
    create_animation(con_mat, Freq_Bands.keys(), epochs.ch_names)

    return con_time

    
