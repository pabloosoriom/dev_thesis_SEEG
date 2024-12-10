import mlflow
import mlflow.pyfunc
import mne
import numpy as np
import time
import json
import os
from collections import defaultdict
import gc
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

warnings.filterwarnings('ignore')

from functions.Connectivity import *
from functions.preprocessing import *
from functions.temporal_networks import *

def run_experiment(patient, output_path,raw, xyz_loc,bands, method_exp, norm, algorithms, inside_network, detail):

    mlflow.set_experiment('Community_Detection_Experiment_vs2')
    
    with mlflow.start_run(run_name=f'Experiment_{patient}_{method_exp}_{detail}',nested=True):

        # Track execution time
        start_time = time.time()
        # Log patient name as a parameter
        mlflow.log_param('Patient', patient)
        results_band_communities_after = defaultdict(dict)
        results_band_communities_before = defaultdict(dict)
        results_band_metrics_after = defaultdict(dict)
        results_band_metrics_before = defaultdict(dict)

        for band in bands:
            results_norm_communities_after = {}
            results_norm_metrics_after = {}
            results_norm_communities_before = {}
            results_norm_metrics_before = {}

            for n in norm:
                # Community detection
                results_algorithm_communities_after = {}
                results_algorithm_communities_before = {}
                results_algorithm_metrics_after = {}
                results_algorithm_metrics_before = {}

                conn = np.load(output_path + f'{patient}_connectivity_data_{band}_{method_exp}_{n}dense.npy')

                
                for algorithm in algorithms:
                    print(f'Running: Band={band}, Method={method_exp}, Normalization={n}, Algorithm={algorithm}')
                    with mlflow.start_run(run_name=f'Algorithm_{algorithm}',nested=True):
                        mlflow.log_param('Band', band)
                        mlflow.log_param('Method', method_exp)
                        mlflow.log_param('Normalization', n)
                        mlflow.log_param('Algorithm', algorithm)

                        
                        if algorithm in ['edge_current_flow_betweenness_partition', 'k_clique_communities']:
                            for k in [2, 3, 4]:
                                communities_before, communities_after, tnet = detect_communities(
                                    conn, xyz_loc=xyz_loc, raw=raw, 
                                    output_path=output_path, threshold_level=0.10, algorithm=algorithm, k=k
                                )


                                #Find the best communities according to density score with regularization 
                                final_communities_before=get_final_communities(communities_before,raw,tnet)
                                final_communities_after=get_final_communities(communities_after,raw,tnet)

                                #Getting the jaccard metric for the best communities
                                jaccard_final_communities_before=jaccard_final_communities(final_communities_before,inside_network)
                                jaccard_final_communities_after=jaccard_final_communities(final_communities_after,inside_network)


                                for t, metric_value in jaccard_final_communities_after.items():
                                    mlflow.log_metric(f'Jaccard_After_{k}_{algorithm}', metric_value, step=t)

                                for t, metric_value in jaccard_final_communities_before.items():
                                    mlflow.log_metric(f'Jaccard_Before_{k}_{algorithm}', metric_value, step=t)

                                #Save the results
                                results_algorithm_communities_after[f'{k}_{algorithm}'] = final_communities_before
                                results_algorithm_communities_before[f'{k}_{algorithm}'] = final_communities_after
                                results_algorithm_metrics_after[f'{k}_{algorithm}'] = jaccard_final_communities_after
                                results_algorithm_metrics_before[f'{k}_{algorithm}'] = jaccard_final_communities_before
    
                        else:
                            communities_before, communities_after,tnet = detect_communities(
                                conn, xyz_loc=xyz_loc, raw=raw, 
                                output_path=output_path, threshold_level=0.10, algorithm=algorithm
                            )

                            #Find the best communities according to density score with regularization
                            final_communities_before=get_final_communities(communities_before,raw,tnet)
                            final_communities_after=get_final_communities(communities_after,raw,tnet)

                            #Getting the jaccard metric for the best communities
                            jaccard_final_communities_before=jaccard_final_communities(final_communities_before,inside_network)
                            jaccard_final_communities_after=jaccard_final_communities(final_communities_after,inside_network)

                            for t, metric_value in jaccard_final_communities_after.items():
                                mlflow.log_metric(f'Jaccard_After_{algorithm}', metric_value, step=t)

                            for t, metric_value in jaccard_final_communities_before.items():
                                mlflow.log_metric(f'Jaccard_Before_{algorithm}', metric_value, step=t)


 
                            results_algorithm_communities_after[algorithm] = final_communities_after
                            results_algorithm_communities_before[algorithm] = final_communities_before
                            results_algorithm_metrics_after[algorithm] = jaccard_final_communities_after
                            results_algorithm_metrics_before[algorithm] = jaccard_final_communities_before
 
                results_norm_communities_after[n] = results_algorithm_communities_after
                results_norm_communities_before[n] = results_algorithm_communities_before
                results_norm_metrics_after[n] = results_algorithm_metrics_after
                results_norm_metrics_before[n] = results_algorithm_metrics_before
            results_band_communities_after[band] = results_norm_communities_after
            results_band_communities_before[band] = results_norm_communities_before
            results_band_metrics_after[band] = results_norm_metrics_after
            results_band_metrics_before[band] = results_norm_metrics_before


        # Log artifacts (JSON files with the results)
        with open(output_path + f'results_communities_after_{method_exp}.json', 'w') as f:
            json.dump(results_band_communities_after, f)
        with open(output_path + f'results_communities_before_{method_exp}.json', 'w') as f:
            json.dump(results_band_communities_before, f)
        with open(output_path + f'results_metrics_after_{method_exp}.json', 'w') as f:
            json.dump(results_band_metrics_after, f)
        with open(output_path + f'results_metrics_before_{method_exp}.json', 'w') as f:
            json.dump(results_band_metrics_before, f)

        mlflow.log_artifact(output_path + f'results_communities_after_{method_exp}.json')
        mlflow.log_artifact(output_path + f'results_communities_before_{method_exp}.json')
        mlflow.log_artifact(output_path + f'results_metrics_after_{method_exp}.json')
        mlflow.log_artifact(output_path + f'results_metrics_before_{method_exp}.json')

        # Log execution time
        end_time = time.time()
        execution_time = end_time - start_time
        mlflow.log_metric('Execution_Time', execution_time)

def main():
    patients = ['pte_01']  # Add more patients here

    inside_network = ["m'3","sc'3","sc'4","sc'5","sc'6","y'4","y'5","y'6","y'7","y'8","y'9"]
    for patient in patients:
        output_path = f'/home/pablo/works/dev_thesis_SEEG/outputs/{patient}/'
        input_path = f'/home/pablo/works/dev_thesis_SEEG/data/{patient}/sets/segments_ictal_SR/'
        xyz_loc_path = f'/home/pablo/works/dev_thesis_SEEG/data/{patient}/others/sEEG_locs.tsv'
        
        # Reading and preprocessing data
        raw = mne.io.read_raw_fif(input_path + 'pte01_sub_9324.fif', preload=True)
        xyz_loc = pd.read_csv(xyz_loc_path, sep='\t')
        raw, xyz_loc = format_data(raw, xyz_loc)
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
            animation=False)
        
        del raw_filtered1, raw_filtered, epochs
        gc.collect()
        
        bands = ['low_gamma', 'high_gamma1']
        method_exp = 'aec'
        norm = ['distance_']
        algorithms = [
            'girvan_newman', 'k_clique_communities',
        ]
    
        run_experiment(patient, output_path, raw, xyz_loc, bands, method_exp, norm, algorithms, inside_network,detail='inter-ictal_epoch3s')

if __name__ == "__main__":
    main()
