import mlflow
import mlflow.pyfunc
import mne
import numpy as np
import time
import json
from collections import defaultdict
import gc
import warnings
import matplotlib
import scipy.io as sio

matplotlib.use('Agg')  # Use non-interactive backend

warnings.filterwarnings('ignore')

from functions.Connectivity import *
from functions.preprocessing import *
from functions.temporal_networks import *
from functions.plotter_comunities import plot_electrode_locations



def main():
    patients = ['pte_02']  # Add more patients here

    # inside_network = ["m'3","sc'3","sc'4","sc'5","sc'6","y'4","y'5","y'6","y'7","y'8","y'9"]
    inside_network=["cp'1","cp'2","cp'3","os'1","os'2","os'3","lp'1", "lp'2"]
    for patient in patients:
        output_path = f'/home/pablo/works/dev_thesis_SEEG/outputs/{patient}/'
        input_path = f'/home/pablo/works/dev_thesis_SEEG/data/{patient}/segments/'
        xyz_loc_path = f'/home/pablo/works/dev_thesis_SEEG/data/{patient}/others/channel.mat'
        
        
        # Reading and preprocessing data
        raw = mne.io.read_raw_fif(input_path + 'pte02_sub_5886.fif', preload=True)

        #XYZ location
        #Pat 01
        # xyz_loc = pd.read_csv(xyz_loc_path, sep='\t')   

        #Pat 02
        xyz = sio.loadmat(xyz_loc_path) 
        xyz=xyz['Channel']
        xyz_loc=pd.DataFrame(xyz[0])



        raw, xyz_loc = format_data(raw, xyz_loc)
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
        Define frequency bands
        # Create the connectivity animation
        create_connectivity(
            epochs=epochs,
            output_path=output_path + patient + '_',
            method=method,
            xyz_loc=xyz_loc,
            animation=True)
        
        del raw_filtered1, raw_filtered, epochs
        gc.collect()
        
        bands = ['theta','alpha','beta','low_gamma', 'high_gamma1']
        method_exp = 'plv'
        norm = ['','distance_']
        algorithms = [
           'k_clique_communities','girvan_newman', 'edge_current_flow_betweenness_partition', 'greedy_modularity_communities', 'louvain_communities','kernighan_lin_bisection'
        ]
        #,'girvan_newman', 'edge_current_flow_betweenness_partition', 'greedy_modularity_communities', 'louvain_communities','kernighan_lin_bisection'
        percentile=90
    
        run_experiment(patient, output_path, raw, xyz_loc, bands, method_exp, norm, algorithms,inside_network,percentile,detail='ictal_sub5886_epoch3s_many_algorithms_percentile_90_plv')
        load_and_plot_community_data(xyz_loc=xyz_loc, output_path=output_path, inside_network=inside_network, bands=bands, norm_settings=norm, method_exp=method_exp)
        print(f'Finished processing patient {patient}')



def run_experiment(patient, output_path,raw, xyz_loc,bands, method_exp, norm, algorithms, inside_network,percentile, detail):

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

                tuples_compiled_after = {}
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
                                    output_path=output_path, threshold_level=percentile, algorithm=algorithm, k=k
                                )


                                #Find the best communities according to density score with regularization 
                                final_communities_before,final_com_time_before=get_final_communities(communities_before,raw,tnet)
                                final_communities_after,final_com_time_after=get_final_communities(communities_after,raw,tnet)

                                #Getting the jaccard metric for the best communities
                                jaccard_final_communities_before=jaccard_final_communities(final_communities_before,inside_network)
                                jaccard_final_communities_after=jaccard_final_communities(final_communities_after,inside_network)


                                for t, metric_value in jaccard_final_communities_after.items():
                                    mlflow.log_metric(f'Jaccard_After_{k}_{algorithm}_{method_exp}', metric_value, step=t)

                                for t, metric_value in jaccard_final_communities_before.items():
                                    mlflow.log_metric(f'Jaccard_Before_{k}_{algorithm}_{method_exp}', metric_value, step=t)

                                # Save the results
                                results_algorithm_communities_after[f'{k}_{algorithm}'] = final_communities_before
                                results_algorithm_communities_before[f'{k}_{algorithm}'] = final_communities_after
                                tuples_compiled_after[algorithm] = final_com_time_after


                                results_algorithm_metrics_after[f'{k}_{algorithm}'] = jaccard_final_communities_after
                                results_algorithm_metrics_before[f'{k}_{algorithm}'] = jaccard_final_communities_before
    
                        else:
                            communities_before, communities_after,tnet = detect_communities(
                                conn, xyz_loc=xyz_loc, raw=raw, 
                                output_path=output_path, threshold_level=threshold_magnitude, algorithm=algorithm
                            )

                            #Find the best communities according to density score with regularization
                            final_communities_before,final_com_time_before=get_final_communities(communities_before,raw,tnet)
                            final_communities_after,final_com_time_after=get_final_communities(communities_after,raw,tnet)


                            # #Getting the jaccard metric for the best communities
                            jaccard_final_communities_before=jaccard_final_communities(final_communities_before,inside_network)
                            jaccard_final_communities_after=jaccard_final_communities(final_communities_after,inside_network)

                            for t, metric_value in jaccard_final_communities_after.items():
                                mlflow.log_metric(f'Jaccard_After_{algorithm}_{method_exp}', metric_value, step=t)

                            for t, metric_value in jaccard_final_communities_before.items():
                                mlflow.log_metric(f'Jaccard_Before_{algorithm}_{method_exp}', metric_value, step=t)


 
                            results_algorithm_communities_after[algorithm] = final_communities_after
                            results_algorithm_communities_before[algorithm] = final_communities_before
                            tuples_compiled_after[algorithm] = final_com_time_after


                            results_algorithm_metrics_after[algorithm] = jaccard_final_communities_after
                            results_algorithm_metrics_before[algorithm] = jaccard_final_communities_before
                        
                #######Finding the best array for the community representation, to get only one final time-series of communities with the ones with the higher density score

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

        mlflow.log_artifact(output_path + f'results_communities_after_{method_exp}.json')
        mlflow.log_artifact(output_path + f'results_communities_before_{method_exp}.json')
        mlflow.log_artifact(output_path + f'results_metrics_after_{method_exp}.json')
        mlflow.log_artifact(output_path + f'results_metrics_before_{method_exp}.json')

        #Finding the best array 


        # Log execution time
        end_time = time.time()
        execution_time = end_time - start_time
        mlflow.log_metric('Execution_Time', execution_time)     


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
