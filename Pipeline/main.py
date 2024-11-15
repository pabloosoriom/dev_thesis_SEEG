import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from nltools.data import Brain_Data, Design_Matrix, Adjacency
import networkx as nx
from scipy import signal
from mne_connectivity import spectral_connectivity_epochs
import seaborn as sns
import scipy.io as sio

from Exploration.EpiIndex import *
from functions.Connectivity import *
from functions.preprocessing import *
from functions.temporal_networks import *


import sys
import time

mne.set_config('MNE_MEMMAP_MIN_SIZE', '10M') 
mne.set_config('MNE_CACHE_DIR', '/dev/shm')

def main():


    ######################Reading data###################
    # Start measuring execution time
    start_time = time.time()

    # Redirect output to a file
    patient = 'pte_02'
    output_path = '/home/pablo/works/dev_thesis_SEEG//outputs/'+patient+'/'
    input_path = '/home/pablo/works/dev_thesis_SEEG/data/'+patient+'/'+'segments/'

    
    
    # # #Reading the signal data
    raw = mne.io.read_raw_fif(input_path + 'pte02_sub_6249' + '.fif', preload=True)

    # #Reading xyz schema
    # xyz_loc = pd.read_csv('/home/pablo/works/dev_thesis_SEEG/data/'+patient+'/'+'sEEG_locs.csv',sep='\t')
    xyz = sio.loadmat('/home/pablo/works/dev_thesis_SEEG/data/pte_02/channel.mat') 
    xyz=xyz['Channel']
    xyz_loc=pd.DataFrame(xyz[0])

    # # # Formating both files to have according channels 

    raw , xyz_loc = format_data(raw, xyz_loc)

    plot_xyz(xyz_loc,outputpath=output_path,label='formatted_label')


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
        animation=True
    )

    # ###################Community Detection###################
    # # Community detection
    # bands=['theta','alpha','beta','low_gamma','high_gamma1']
    # methods=['aec','plv']
    # norm=['distance_','']
    # algorithms= ['girvan_newman','edge_current_flow_betweenness_partition','k_clique_communities','naive_greedy_modularity_communities','kernighan_lin_bisection']

    # results_communities = []
    # results_metrics = []
    # results_band_communities_after = {}
    # results_band_communities_before = {}

    # results_band_metrics_after = {}
    # results_band_metrics_before = {}
    # for band in bands:
    #     results_method_communities_after = {}
    #     results_method_communities_before = {}
    #     results_method_metrics_after = {}
    #     results_method_metrics_before = {}
    #     for method in methods:
    #         results_norm_communities_after = {}
    #         results_norm_metrics_after = {}

    #         results_norm_communities_before = {}
    #         results_norm_metrics_before = {}
    #         for n in norm:
    #              #Read the connectivity data
    #             conn = np.load(output_path + f'{patient}_connectivity_data_{band}_{method}_{n}dense.npy')

    #             # Community detection
    #             results_algorithm_communities_after = {}
    #             results_algorithm_communities_before = {}

    #             results_algorithm_metrics_after = {}
    #             results_algorithm_metrics_before = {}
    #             for algorithm in algorithms:
    #                 print(f'Band: {band}, Method: {method}, Normalization: {n}, Algorithm: {algorithm}')
    #                 # Load the connectivity data
    #                 conn = Adjacency.load(output_path + f'{patient}_{method}_{band}_connectivity_{n}.nii.gz')
    #                 # Community detection
    #                 k=[2,3,4]
    #                 if algorithm == 'edge_current_flow_betweenness_partition' or algorithm == 'k_clique_communities':
    #                     for k in k:
    #                         communities_before, communities_after = detect_communities(
    #                             conn,
    #                             xyz_loc=xyz_loc,
    #                             raw=raw,
    #                             output_path=output_path,
    #                             threshold_level=0.15,
    #                             algorithm=algorithm,
    #                             k=k)
    #                     #Rebuilding the communities for each time step
    #                     max_jaccard_after,communities_dict_after=jaccard_metric(communities_after)
    #                     max_jaccard_before,communities_dict_before=jaccard_metric(communities_before)

    #                     #Save the results
    #                     results_algorithm_communities_after[k+'_'+algorithm] = communities_dict_after
    #                     results_algorithm_communities_before[k+'_'+algorithm] = communities_dict_before
    #                     results_algorithm_metrics_after[k+'_'+algorithm] = max_jaccard_after
    #                     results_algorithm_metrics_before[k+'_'+algorithm] = max_jaccard_before
    #                 else:
    #                     communities_before, communities_after = detect_communities(
    #                         conn,
    #                         xyz_loc=xyz_loc,
    #                         raw=raw,
    #                         output_path=output_path,
    #                         threshold_level=0.15,
    #                         algorithm=algorithm)
    #                     #Rebuilding the communities for each time step
    #                     max_jaccard_after,communities_dict_after=jaccard_metric(communities_after)
    #                     max_jaccard_before,communities_dict_before=jaccard_metric(communities_before)

    #                     #Save the results
    #                     results_algorithm_communities_after[algorithm] = communities_dict_after
    #                     results_algorithm_communities_before[algorithm] = communities_dict_before
    #                     results_algorithm_metrics_after[algorithm] = max_jaccard_after
    #                     results_algorithm_metrics_before[algorithm] = max_jaccard_before
                
    #             results_norm_communities_after[n] = results_algorithm_communities_after
    #             results_norm_communities_before[n] = results_algorithm_communities_before
    #             results_norm_metrics_after[n] = results_algorithm_metrics_after
    #             results_norm_metrics_before[n] = results_algorithm_metrics_before
            
    #         results_method_communities_after[method] = results_norm_communities_after
    #         results_method_communities_before[method] = results_norm_communities_before
    #         results_method_metrics_after[method] = results_norm_metrics_after
    #         results_method_metrics_before[method] = results_norm_metrics_before

    #     results_band_communities_after[band] = results_method_communities_after
    #     results_band_communities_before[band] = results_method_communities_before
    #     results_band_metrics_after[band] = results_method_metrics_after
    #     results_band_metrics_before[band] = results_method_metrics_before
    
    # #Save the results in json files
    # with open(output_path + 'results_communities_after.json', 'w') as f:
    #     json.dump(results_band_communities_after, f)
    # with open(output_path + 'results_communities_before.json', 'w') as f:
    #     json.dump(results_band_communities_before, f)
    # with open(output_path + 'results_metrics_after.json', 'w') as f:
    #     json.dump(results_band_metrics_after, f)
    # with open(output_path + 'results_metrics_before.json', 'w') as f:
    #     json.dump(results_band_metrics_before, f)

    # End measuring execution time
    end_time = time.time()
    execution_time = end_time - start_time

    print(f'Total execution time: {execution_time} seconds')




if __name__ == "__main__":
    main()
# segment_processing()