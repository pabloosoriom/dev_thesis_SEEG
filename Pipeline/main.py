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


def main():
    # Start measuring execution time
    start_time = time.time()

    # Redirect output to a file
    output_path = '/home/pablo/works/dev_thesis_SEEG/data/outputs/'
    input_path = '/home/pablo/works/dev_thesis_SEEG/data/'
    patient = 'pte_6'

    # # #Filtering
    raw_cleaned = mne.io.read_raw_fif(input_path + patient + '_cleaned.fif', preload=True)
    # # #raw_cleaned, fig = clean_data(raw, 'Cz')
    raw_high_pass, fig1 = high_pass_filter(raw_cleaned)
    raw_low_pass, fig2 = low_pass_filter(raw_high_pass)
    print(raw_low_pass)
    print(raw_low_pass.info)

    # # #Save the figures and the filtered data
    fig1.savefig(output_path + patient + '_high_pass_filter.png')
    fig2.savefig(output_path + patient + '_low_pass_filter.png')
    raw_low_pass = set_names(raw_low_pass)
    raw_low_pass.save(output_path + patient + '_filtered.fif', overwrite=True)


    # # # Epoching
    raw = mne.io.read_raw_fif(output_path + patient + '_filtered.fif', preload=True)
    t_sec=raw.n_times/raw.info['sfreq']
    epochs=mne.make_fixed_length_epochs(raw, duration=t_sec/15, preload=True)
    print(epochs)

    # # # Connectivity
    # Define frequency bands
    # Create the connectivity animation
    create_connectivity_animation(
        epochs=epochs,
        output_path=output_path + patient + '_'
    )

    # print('Problematic channels dropped from the main database')

    # channels = raw.ch_names

    # Ei_n1, ER_matrix1, U_n_matrix, ER_n_array, alarm_time = get_EI(raw)

    # plotting_ei(Ei_n1, ER_matrix1, channels, save_path='.lightning_studio/EpiPlan /Pipeline/outputs')

    # # Save Ei_n1 matrix to CSV
    # np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/Ei_n1.csv', Ei_n1, delimiter=';')

    # # Save ER_matrix1 matrix to CSV
    # np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/ER_matrix1.csv', ER_matrix1, delimiter=';')

    # # Save U_n_matrix matrix to CSV
    # np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/U_n_matrix.csv', U_n_matrix, delimiter=';')

    # # Save ER_n_array matrix to CSV
    # np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/ER_n_array.csv', ER_n_array, delimiter=';')

    # # Save alarm_time matrix to CSV
    # np.savetxt('.lightning_studio/EpiPlan/Pipeline/outputs/alarm_time.csv', alarm_time, delimiter=';')
    # # Stop measuring execution time
    end_time = time.time()
    execution_time = end_time - start_time

    print(f'Total execution time: {execution_time} seconds')

    # # Print and save the execution time
    # print(f'Total execution time: {execution_time} seconds')
    # with open('.lightning_studio/EpiPlan/Pipeline/output.txt', 'a') as f:
    #     f.write(f'\nTotal execution time: {execution_time} seconds\n')

    # Close the redirected output file
    # sys.stdout.close()






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