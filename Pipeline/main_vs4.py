import mne
import numpy as np
import time
import json
from collections import defaultdict
import gc
import warnings
import matplotlib
import scipy.io as sio
import os

matplotlib.use('Agg')  # Use non-interactive backend

warnings.filterwarnings('ignore')

from functions.Connectivity import *
from functions.preprocessing import *
from functions.temporal_networks import *
from functions.plotter_comunities import plot_electrode_locations



def main():
    patients = ['sub-HUP117','sub-HUP130','sub-HUP134','sub-HUP139','sub-HUP140','sub-HUP141','sub-HUP144']  #Patients to process
    
    ref_data=0

    for patient in patients:
        print(f'Processing patient {patient} in file {ref_data}')
        #Check if there is a folder for the patient. If not, create it
        if not os.path.exists(f'/home/pablo/works/dev_thesis_SEEG/outputs/{patient}/'):
            os.makedirs(f'/home/pablo/works/dev_thesis_SEEG/outputs/{patient}/')

        if not os.path.exists(f'/home/pablo/works/dev_thesis_SEEG/outputs/{patient}/ref_{ref_data}/'):
            os.makedirs(f'/home/pablo/works/dev_thesis_SEEG/outputs/{patient}/ref_{ref_data}/')
        
        output_path = f'/home/pablo/works/dev_thesis_SEEG/outputs/{patient}/ref_{ref_data}/'



        #Define input paths
        main_path='/home/pablo/works/dev_thesis_SEEG/data/mainDatabase_patients/'
        # Document with file data
        doc_file_data=pd.read_csv(main_path+'/'+patient+'/'+'ses-presurgery/'+patient+'_ses-presurgery_scans.tsv',sep='\t')
        raw=mne.io.read_raw_edf(main_path+'/'+patient+'/'+'ses-presurgery/'+doc_file_data['filename'][ref_data],preload=True)
        xyz_loc=pd.read_csv(main_path+'/'+patient+'/'+'ses-presurgery/ieeg/'+patient+'_ses-presurgery_acq-seeg_space-fsaverage_electrodes.tsv',sep='\t')
        events=pd.read_csv(main_path+'/'+patient+'/'+'ses-presurgery/'+doc_file_data['filename'][ref_data].replace('_ieeg.edf','_events.tsv'),sep='\t')
        channels=pd.read_csv(main_path+'/'+patient+'/'+'ses-presurgery/'+doc_file_data['filename'][ref_data].replace('_ieeg.edf','_channels.tsv'),sep='\t')


        raw, xyz_loc, inside_network= format_data(raw, xyz_loc, events, channels)

        #save the xyz_loc
        xyz_loc.to_csv(output_path + 'xyz_loc.csv', sep='\t', index=False)
        

        raw_filtered1, _ = pass_filter(raw)
        raw_filtered, _ = line_noise_filter(raw_filtered1, [60, 120], True)
        raw_filtered = set_names(raw_filtered)
        raw_filtered.save(output_path + patient + '_filtered.fif', overwrite=True)

        # Load filtered data
        raw = mne.io.read_raw_fif(output_path + patient + '_filtered.fif', preload=True)
        epochs = mne.make_fixed_length_epochs(raw, duration=3, preload=True)
        bad_epochs = tag_high_amplitude(epochs)

        if bad_epochs:
            epochs.drop(bad_epochs)

        # Connectivity
        method = 'aec&plv'
        print(f'Connectivity method: {method}')
        # Define frequency bands
        # Create the connectivity animation
        create_connectivity(
            epochs=epochs,
            output_path=output_path + patient + '_',
            method=method,
            xyz_loc=xyz_loc,
            animation=True)
        
        del raw_filtered1, raw_filtered, epochs
        gc.collect()
        
        bands = ['low_gamma','high_gamma1']
        method_exp = 'aec'
        norm = ['','distance_']
        algorithms = [
           'k_clique_communities','girvan_newman', 'edge_current_flow_betweenness_partition', 'greedy_modularity_communities', 'louvain_communities','kernighan_lin_bisection'
        ]
        #'k_clique_communities','girvan_newman', 'edge_current_flow_betweenness_partition', 'greedy_modularity_communities', 'louvain_communities','kernighan_lin_bisection'
        percentile=0.90

        print(f'Processing patient {patient} with percentile {percentile}')
    
        run_experiment(patient, output_path, raw, xyz_loc, bands, method_exp, norm, algorithms,inside_network,percentile,detail='ictal_sub000_epoch3s_many_algorithms_percentile_90_plv')
        load_and_plot_community_data(xyz_loc=xyz_loc, output_path=output_path, inside_network=inside_network, bands=bands, norm_settings=norm, method_exp=method_exp)
        print(f'Finished processing patient {patient}')



def run_experiment(patient, output_path,raw, xyz_loc,bands, method_exp, norm, algorithms, inside_network,percentile, detail):
    # Track execution time
    start_time = time.time()
    # Log patient name as a parameter
    results_band_communities_after = defaultdict(dict)
    results_band_communities_before = defaultdict(dict)
    results_band_metrics_after = defaultdict(dict)
    results_band_metrics_before = defaultdict(dict)

    final_com_time_after_dict= {}

    for band in bands:
        results_norm_communities_after = {}
        results_norm_metrics_after = {}
        results_norm_communities_before = {}
        results_norm_metrics_before = {}

        final_com_norm_after = {}

        for n in norm:
            # Community detection
            results_algorithm_communities_after = {}
            results_algorithm_communities_before = {}
            results_algorithm_metrics_after = {}
            results_algorithm_metrics_before = {}

            conn = np.load(output_path + f'{patient}_connectivity_data_{band}_{method_exp}_{n}dense.npy')
            # Threshold detection
            threshold_magnitude= plot_weight_distribution_with_threshold(conn,output_path,band,bins=30,percentile=percentile, method_exp=method_exp)
            print(f'Mean threshold magnitude: {threshold_magnitude}')

            tuples_compiled_after = {}
            for algorithm in algorithms:
                print(f'Running: Band={band}, Method={method_exp}, Normalization={n}, Algorithm={algorithm}')
                                
                if algorithm in ['edge_current_flow_betweenness_partition', 'k_clique_communities']:
                    for k in [2, 3, 4]:
                        communities_before, communities_after, tnet = detect_communities(
                            conn, xyz_loc=xyz_loc, raw=raw, 
                            output_path=output_path, threshold_level_=percentile, algorithm=algorithm, k=k
                        )


                        #Find the best communities according to density score with regularization 
                        final_communities_before,final_com_time_before=get_final_communities(communities_before,raw,tnet)
                        final_communities_after,final_com_time_after=get_final_communities(communities_after,raw,tnet)

                        #Getting the jaccard metric for the best communities
                        jaccard_final_communities_before=jaccard_final_communities(final_communities_before,inside_network)
                        jaccard_final_communities_after=jaccard_final_communities(final_communities_after,inside_network)

                        # Save the results
                        results_algorithm_communities_after[f'{k}_{algorithm}'] = final_communities_before
                        results_algorithm_communities_before[f'{k}_{algorithm}'] = final_communities_after
                        tuples_compiled_after[f'{k}_{algorithm}'] = final_com_time_after


                        results_algorithm_metrics_after[f'{k}_{algorithm}'] = jaccard_final_communities_after
                        results_algorithm_metrics_before[f'{k}_{algorithm}'] = jaccard_final_communities_before

                else:
                    communities_before, communities_after,tnet = detect_communities(
                        conn, xyz_loc=xyz_loc, raw=raw, 
                        output_path=output_path, threshold_level_=percentile, algorithm=algorithm
                    )

                    #Find the best communities according to density score with regularization
                    final_communities_before,final_com_time_before=get_final_communities(communities_before,raw,tnet)
                    final_communities_after,final_com_time_after=get_final_communities(communities_after,raw,tnet)


                    # #Getting the jaccard metric for the best communities
                    jaccard_final_communities_before=jaccard_final_communities(final_communities_before,inside_network)
                    jaccard_final_communities_after=jaccard_final_communities(final_communities_after,inside_network)

                    results_algorithm_communities_after[algorithm] = final_communities_after
                    results_algorithm_communities_before[algorithm] = final_communities_before
                    tuples_compiled_after[algorithm] = final_com_time_after


                    results_algorithm_metrics_after[algorithm] = jaccard_final_communities_after
                    results_algorithm_metrics_before[algorithm] = jaccard_final_communities_before
                    
            #######Finding the best array for the community representation, to get only one final time-series of communities with the ones with the higher density score
            #Save the final time-series of communities
            with open(output_path + f'Tuples_com_{band}_{method_exp}_{n}.json', 'w') as f:
                json.dump(tuples_compiled_after, f)


            final_time_community= get_max_communities(tuples_compiled_after)
            #Save the final time-series of communities
            with open(output_path + f'final_com_{band}_{method_exp}_{n}.json', 'w') as f:
                json.dump(final_time_community, f)
            
            #Getting final metrics for the best communities for experiment


            del conn


            gc.collect()
            results_norm_communities_after[n] = results_algorithm_communities_after
            results_norm_communities_before[n] = results_algorithm_communities_before
            results_norm_metrics_after[n] = results_algorithm_metrics_after
            results_norm_metrics_before[n] = results_algorithm_metrics_before

            final_com_norm_after[n] = tuples_compiled_after


        results_band_communities_after[band] = results_norm_communities_after
        results_band_communities_before[band] = results_norm_communities_before
        results_band_metrics_after[band] = results_norm_metrics_after
        results_band_metrics_before[band] = results_norm_metrics_before

        final_com_time_after_dict[band] = final_com_norm_after


    # Log artifacts (JSON files with the results)
    with open(output_path + f'results_communities_after_{method_exp}.json', 'w') as f:
        json.dump(results_band_communities_after, f)
    with open(output_path + f'results_communities_before_{method_exp}.json', 'w') as f:
        json.dump(results_band_communities_before, f)
    with open(output_path + f'results_metrics_after_{method_exp}.json', 'w') as f:
        json.dump(results_band_metrics_after, f)
    with open(output_path + f'results_metrics_before_{method_exp}.json', 'w') as f:
        json.dump(results_band_metrics_before, f)

    with open(output_path + f'final_com_time_after_{method_exp}.json', 'w') as f:
        json.dump(final_com_time_after_dict, f)
    # Log execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Execution time: {execution_time} seconds')


def load_and_plot_community_data(xyz_loc, output_path, inside_network, bands, norm_settings,method_exp='aec'):
    """
    Loads community data files, plots electrode locations, and final metrics for each specified band and normalization setting.

    Parameters:
    - xyz_loc: DataFrame or dict with electrode coordinates and labels.
    - output_path: str, path to directory containing community data files.
    - inside_network: data structure required for final_metrics_plot.
    - bands: list of str, each representing a frequency band (e.g., ['high_gamma1', 'low_gamma', 'alpha', 'beta', 'theta']).
    - norm_settings: list of str, normalization settings (e.g., ['distance_', '']).

    """
    for band in bands:
        for norm in norm_settings:
            file_suffix = f"{band}_{method_exp}_{norm}.json"
            try:
                # Load community data
                with open(f"{output_path}final_com_{file_suffix}") as f:
                    communities_data = json.load(f)

                # Determine normalization label for plot
                norm_label = "no-norm" if norm == "" else norm.strip('_')

                # Plot electrode locations and metrics
                plot_electrode_locations(
                    xyz_loc=xyz_loc,
                    communities_data=communities_data,
                    outputpath=output_path,
                    band=f"{band}_{norm_label}_{method_exp}"
                )
                final_metrics_plot(
                    communities_data=communities_data,
                    inside_networks=inside_network,
                    output_path=output_path,
                    band=f"{band}_{norm_label}_{method_exp}"
                )

                print(f"Processed {file_suffix}")
                
            except FileNotFoundError:
                print(f"File not found: {output_path}final_com_{file_suffix}")



if __name__ == "__main__":
    main()
