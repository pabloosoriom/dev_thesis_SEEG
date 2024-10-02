import mlflow
import mlflow.pyfunc
import mne
import numpy as np
import time
import json
import os
from collections import defaultdict
import gc


from functions.EpiIndex import *
from functions.Connectivity import *
from functions.preprocessing import *
from functions.temporal_networks import *

def run_experiment(patient, output_path, input_path, xyz_loc_path, bands, methods, norm, algorithms, inside_network):

    mlflow.set_experiment('Community_Detection_Experiment')
    
    with mlflow.start_run(run_name=f'Experiment_{patient}',nested=True):
        # Log patient name as a parameter
        mlflow.log_param('Patient', patient)

        # Track execution time
        start_time = time.time()

        # Reading and preprocessing data
        raw = mne.io.read_raw_fif(input_path + 'ictal-epo_6.fif', preload=True)
        xyz_loc = pd.read_csv(xyz_loc_path, sep='\t')
        raw, xyz_loc = format_data(raw, xyz_loc)
        raw_filtered1, _ = pass_filter(raw)
        raw_filtered, _ = line_noise_filter(raw_filtered1, [60, 120], True)
        raw_filtered = set_names(raw_filtered)
        raw_filtered.save(output_path + patient + '_filtered.fif', overwrite=True)

        # Load filtered data
        raw = mne.io.read_raw_fif(output_path + patient + '_filtered.fif', preload=True)
        epochs = mne.make_fixed_length_epochs(raw, duration=30, preload=True)
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
        
        

        results_band_communities_after = defaultdict(dict)
        results_band_communities_before = defaultdict(dict)
        results_band_metrics_after = defaultdict(dict)
        results_band_metrics_before = defaultdict(dict)

        for band in bands:
            results_method_communities_after = {}
            results_method_communities_before = {}
            results_method_metrics_after = {}
            results_method_metrics_before = {}

            for method in methods:
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

                    conn = np.load(output_path + f'{patient}_connectivity_data_{band}_{method}_{n}dense.npy')

                    
                    for algorithm in algorithms:
                        print(f'Running: Band={band}, Method={method}, Normalization={n}, Algorithm={algorithm}')
                        with mlflow.start_run(run_name=f'Algorithm_{algorithm}',nested=True):
                            mlflow.log_param('Band', band, 'Method', method, 'Normalization', n, 'Algorithm', algorithm)
                           
                            if algorithm in ['edge_current_flow_betweenness_partition', 'k_clique_communities']:
                                for k in [2, 3, 4]:
                                    communities_before, communities_after = detect_communities(
                                        conn, xyz_loc=xyz_loc, raw=raw, 
                                        output_path=output_path, threshold_level=0.10, algorithm=algorithm, k=k
                                    )
                                    max_jaccard_after, communities_dict_after = jaccard_metric(communities_after,raw,inside_network)
                                    max_jaccard_before, communities_dict_before = jaccard_metric(communities_before,raw,inside_network)
                                    
                                    # Log metrics
                                    # Batch log metrics to reduce `mlflow` overhead
                                    mlflow.log_metrics({f'Jaccard_After_{k}_{algorithm}_{idx}': value for idx, value in enumerate(max_jaccard_after)})
                                    mlflow.log_metrics({f'Jaccard_Before_{k}_{algorithm}_{idx}': value for idx, value in enumerate(max_jaccard_before)})

                                    # mlflow.log_metric(f'Jaccard_After_{k}_{algorithm}', max_jaccard_after)
                                    # mlflow.log_metric(f'Jaccard_Before_{k}_{algorithm}', max_jaccard_before)

                                    #Save the results
                                    results_algorithm_communities_after[f'{k}_{algorithm}'] = communities_dict_after
                                    results_algorithm_communities_before[f'{k}_{algorithm}'] = communities_dict_before
                                    results_algorithm_metrics_after[f'{k}_{algorithm}'] = max_jaccard_after
                                    results_algorithm_metrics_before[f'{k}_{algorithm}'] = max_jaccard_before
                                      # Store results directly without creating separate variables
                                    # results_norm_communities_after[f'{k}_{algorithm}'] = communities_dict_after
                                    # results_norm_communities_before[f'{k}_{algorithm}'] = communities_dict_before
                                    # results_norm_metrics_after[f'{k}_{algorithm}'] = max_jaccard_after
                                    # results_norm_metrics_before[f'{k}_{algorithm}'] = max_jaccard_before
                            else:
                                communities_before, communities_after = detect_communities(
                                    conn, xyz_loc=xyz_loc, raw=raw, 
                                    output_path=output_path, threshold_level=0.10, algorithm=algorithm
                                )

                                max_jaccard_after, communities_dict_after = jaccard_metric(communities_after,raw, inside_network)
                                max_jaccard_before, communities_dict_before = jaccard_metric(communities_before,raw, inside_network)
                                mlflow.log_metrics({f'Jaccard_After_{algorithm}_{idx}': value for idx, value in enumerate(max_jaccard_after)})
                                mlflow.log_metrics({f'Jaccard_Before_{algorithm}_{idx}': value for idx, value in enumerate(max_jaccard_before)})


                                # max_jaccard_after, communities_dict_after = jaccard_metric(communities_after,raw, inside_network)
                                # max_jaccard_before, communities_dict_before = jaccard_metric(communities_before,raw, inside_network)
                                #Save the results
                                results_algorithm_communities_after[algorithm] = communities_dict_after
                                results_algorithm_communities_before[algorithm] = communities_dict_before
                                results_algorithm_metrics_after[algorithm] = max_jaccard_after
                                results_algorithm_metrics_before[algorithm] = max_jaccard_before
                                # results_norm_communities_after[algorithm] = communities_dict_after
                                # results_norm_communities_before[algorithm] = communities_dict_before
                                # results_norm_metrics_after[algorithm] = max_jaccard_after
                                # results_norm_metrics_before[algorithm] = max_jaccard_before

                        gc.collect()

                                # Log metrics
                                # mlflow.log_metric(f'Jaccard_After_{algorithm}', max_jaccard_after)
                                # mlflow.log_metric(f'Jaccard_Before_{algorithm}', max_jaccard_before)
                    results_norm_communities_after[n] = results_algorithm_communities_after
                    results_norm_communities_before[n] = results_algorithm_communities_before
                    results_norm_metrics_after[n] = results_algorithm_metrics_after
                    results_norm_metrics_before[n] = results_algorithm_metrics_before
                results_method_communities_after[method] = results_norm_communities_after
                results_method_communities_before[method] = results_norm_communities_before
                results_method_metrics_after[method] = results_norm_metrics_after
                results_method_metrics_before[method] = results_norm_metrics_before
            results_band_communities_after[band] = results_method_communities_after
            results_band_communities_before[band] = results_method_communities_before
            results_band_metrics_after[band] = results_method_metrics_after
            results_band_metrics_before[band] = results_method_metrics_before



        # Log artifacts (JSON files with the results)
        with open(output_path + 'results_communities_after.json', 'w') as f:
            json.dump(results_band_communities_after, f)
        with open(output_path + 'results_communities_before.json', 'w') as f:
            json.dump(results_band_communities_before, f)
        with open(output_path + 'results_metrics_after.json', 'w') as f:
            json.dump(results_band_metrics_after, f)
        with open(output_path + 'results_metrics_before.json', 'w') as f:
            json.dump(results_band_metrics_before, f)

        mlflow.log_artifact(output_path + 'results_communities_after.json')
        mlflow.log_artifact(output_path + 'results_communities_before.json')
        mlflow.log_artifact(output_path + 'results_metrics_after.json')
        mlflow.log_artifact(output_path + 'results_metrics_before.json')

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
        
        bands = ['low_gamma', 'high_gamma1']
        methods = ['aec', 'plv']
        norm = ['distance_']
        algorithms = [
            'girvan_newman', 'k_clique_communities'
        ]
    
        run_experiment(patient, output_path, input_path, xyz_loc_path, bands, methods, norm, algorithms, inside_network)

if __name__ == "__main__":
    main()
