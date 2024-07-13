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


import sys

def main():
   import sys
import time

def main():
    # Start measuring execution time
    start_time = time.time()

    # Redirect output to a file
    sys.stdout = open('.lightning_studio/EpiPlan /Pipeline/outputs/output.txt', 'w')

    data = mne.io.read_raw_edf("/teamspace/studios/this_studio/.lightning_studio/EpiPlan /data/pte6_1_28_38.edf", preload=True, infer_types=True)
    print(f"Total data: {data.n_times}, with a total time of {data.n_times/data.info['sfreq']} seconds")
    raw=data.copy().crop(5300, 5900)
    print(f"Data reduced to: {raw.n_times} samples, with a total time of {raw.n_times/raw.info['sfreq']} seconds")

    # Get the channel names
    ch_names = raw.ch_names

    # Dictionary to hold channel types
    channel_types = {}

    # Set all channel types to 'seeg'
    for ch_name in ch_names:
        channel_types[ch_name] = 'seeg'

    # Set the channel types
    raw.set_channel_types(channel_types)

    # Verify that the channel types are correctly set
    print(raw.info)

    # Using a reference channel
    reference_channel = 'MKR2+'
    correlation_values = np.corrcoef(raw[reference_channel][0], raw.get_data())[0, 1:]
    correlation_threshold = 0.4
    outlier_channels = np.where(np.abs(correlation_values) < correlation_threshold)[0]

    print(f'Problematic channel 1: {raw.ch_names[64]}')
    print(f'Problematic channel 2: {raw.ch_names[129]}')

    raw.drop_channels([raw.ch_names[64], raw.ch_names[129]])

    print('Problematic channels dropped from the main database')

    channels = raw.ch_names

    Ei_n1, ER_matrix1, U_n_matrix, ER_n_array, alarm_time = get_EI(raw)

    plotting_ei(Ei_n1, ER_matrix1, channels, save_path='.lightning_studio/EpiPlan /Pipeline/outputs')

    # Save Ei_n1 matrix to CSV
    np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/Ei_n1.csv', Ei_n1, delimiter=';')

    # Save ER_matrix1 matrix to CSV
    np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/ER_matrix1.csv', ER_matrix1, delimiter=';')

    # Save U_n_matrix matrix to CSV
    np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/U_n_matrix.csv', U_n_matrix, delimiter=';')

    # Save ER_n_array matrix to CSV
    np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/ER_n_array.csv', ER_n_array, delimiter=';')

    # Save alarm_time matrix to CSV
    np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/alarm_time.csv', alarm_time, delimiter=';')
    # Stop measuring execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Print and save the execution time
    print(f'Total execution time: {execution_time} seconds')
    with open('.lightning_studio/EpiPlan/Pipeline/output.txt', 'a') as f:
        f.write(f'\nTotal execution time: {execution_time} seconds\n')

    # Close the redirected output file
    sys.stdout.close()






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



main()

