import numpy as np
from nltools.data import Adjacency
import networkx as nx
import pandas as pd
from functools import reduce
import teneto
from teneto import plot 
from teneto import TemporalNetwork
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm



#### Graph metrics ####


### Temporal plots ###

def plot_temporal_graph(tnet, output_path, xyz_loc):
    fig,ax = plt.subplots(figsize=(25,15))
    ax = plot.slice_plot(tnet.network, ax, plotedgeweights=False,cmap='Pastel2', nodelabels=list(xyz_loc['formatted_label'].values))
    plt.tight_layout()




### Community detection ###
def detect_communities(data, xyz_loc, raw, output_path, threshold_level, algorithm='k_clique_communities', k=2, plot=False):
    """
    Detects communities over time using the given data and algorithm, and then applies temporal consensus.
    
    Parameters:
    data (numpy array): The input data (3D array) representing adjacency matrices over time.
    xyz_loc (pd.DataFrame): DataFrame containing formatted labels for channels.
    raw (mne.Raw object): Raw data object containing channel names.
    threshold_level (float): Threshold for binarizing the network.
    algorithm (str): Community detection algorithm to use. Options include 'k_clique_communities', 
                     'girvan_newman', 'edge_betweenness_partition', etc.
    k (int): Number of sets or the clique size for k_clique_communities (used in some algorithms).
    
    Returns:
    output (list): Community assignment over time before temporal consensus.
    communities_after (list): Community assignment over time after temporal consensus.
    """
    
    # Transpose data and initialize TemporalNetwork
    data = data.transpose(1, 2, 0)
    tnet_bu = TemporalNetwork(N=data.shape[0], T=data.shape[2], nettype='wu', from_array=data,
                              timetype='discrete', timeunit='epoch', nodelabels=list(xyz_loc['formatted_label'].values))
    
    # Binarize the temporal network
    tnet_bu.binarize(threshold_type='percent', threshold_level=1-threshold_level,axis='graphlet')
    
    # Get the binarized adjacency network
    tnet_bu_ar = tnet_bu.network
    
    # Initialize a dictionary to store communities
    communities_dict = {}
    
    # Detect communities at each time step

    for i in tqdm(range(tnet_bu_ar.shape[2]), desc='Detecting communities over time'):
        # Create a network from the adjacency matrix
        adj = Adjacency(tnet_bu_ar[:, :, i], labels=raw.ch_names)
        G = nx.Graph(adj.to_graph())
   
        # print(f'Detecting communities at time step {i}...')
        
        # Select the community detection algorithm
        communities_dict[i] = communities_algorithm(G, algorithm, k)
    
    # Initialize the output array to store community assignments before consensus
    num_timesteps = len(communities_dict)
    num_channels = len(raw.ch_names)
    comunities_before = [[-1 for _ in range(num_timesteps)] for _ in range(num_channels)]
    
    # Assign communities to the channels
    for t, subcommunities in communities_dict.items():
        for subcom_idx, subcommunity in enumerate(subcommunities):
            for channel in subcommunity:
                if channel in raw.ch_names:
                    channel_idx = raw.ch_names.index(channel)
                    comunities_before[channel_idx][t] = subcom_idx
    
    # Perform temporal consensus
    communities_after = teneto.communitydetection.make_temporal_consensus(comunities_before)

    # Plot the community assignment over time
    if plot:
        plot_community_assignment(communities_after,comunities_before,algorithm,output_path)
    
    print(f'Community detection using {algorithm} completed.')
    
    return comunities_before, communities_after, tnet_bu_ar


def rebuild_communities(communities,raw):
    """
    Rebuilds the communities for each time step.
    
    Parameters:
    communities (list): List of communities, where each community is represented as a list of nodes.
    raw (mne.Raw object): Raw data object containing channel names.
    
    Returns:
    dict: A dictionary where each key is a time step and the value is a dictionary of communities.
    """
    communities_dict = {}
    
    for t in range(len(communities[0])):
        temp_com = {}
        for i, ch in enumerate(raw.ch_names):
            temp_com[communities[i][t]] = temp_com.get(communities[i][t], []) + [ch]
        communities_dict[t] = temp_com
    
    return communities_dict

def communities_algorithm(G, algorithm, k=2):
    """
    Detects communities in the graph G using the specified algorithm.
    
    Parameters:
    G (networkx.Graph): The input graph.
    algorithm (str): The name of the community detection algorithm to use.
    k (int): The number of sets or clique size (used in some algorithms like k_clique_communities).
    
    Returns:
    list: A list of communities, where each community is represented as a list of nodes.
    """
    communities = []

    # Select the community detection algorithm
    if algorithm == 'girvan_newman':
        communities_generator = nx.community.girvan_newman(G)
        next_level_communities = next(communities_generator)
        communities = sorted(map(sorted, next_level_communities))
    
    elif algorithm == 'edge_current_flow_betweenness_partition':
        communities_generator = nx.community.edge_current_flow_betweenness_partition(G, number_of_sets=k)
        communities = [list(c) for c in communities_generator]
    
    elif algorithm == 'edge_betweenness_partition':
        communities_generator = nx.community.edge_betweenness_partition(G, number_of_sets=k)
        communities = [list(c) for c in communities_generator]
    
    elif algorithm == 'k_clique_communities':
        
        cliques = nx.find_cliques(G)
        cliques= [frozenset(c) for c in cliques if len(c) >= k]
        # print(f'Number of cliques {len(cliques)}')     
        if len(cliques) < 20000:
            communities_generator = nx.community.k_clique_communities(G, k)
            communities = [list(c) for c in communities_generator]
        else:
            #all channels
            communities = [list(G.nodes)]
    
    elif algorithm == 'greedy_modularity_communities':
        communities_generator = nx.community.greedy_modularity_communities(G)
        communities = [list(c) for c in communities_generator]
    
    elif algorithm == 'naive_greedy_modularity_communities':
        communities_generator = nx.community.naive_greedy_modularity_communities(G)
        communities = [list(c) for c in communities_generator]
    
    elif algorithm == 'fast_label_propagation_communities':
        communities_generator = nx.community.fast_label_propagation_communities(G)
        communities = [list(c) for c in communities_generator]
    
    elif algorithm == 'louvain_communities':
        communities_generator = nx.community.louvain_communities(G)
        communities = [list(c) for c in communities_generator]
    
    elif algorithm == 'asyn_fluidc':
        # This algorithm only works for undirected and completely connected graphs
        communities_generator = nx.community.asyn_fluidc(G, k)
        communities = [list(c) for c in communities_generator]
    
    elif algorithm == 'kernighan_lin_bisection':
        communities = [list(nx.community.kernighan_lin_bisection(G)[0])]
        communities.append(list(nx.community.kernighan_lin_bisection(G)[1]))
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return communities


def plot_community_assignment(communities_after,communities_before,algorithm,output_path,xyz_loc):
    #Compare the communities
    fig, axes = plt.subplots(1, 2, figsize=(15, 20))

    # Plot for 'Community assignment over time before temporal consensus'
    im1 = axes[0].imshow(communities_before, aspect='auto', cmap='tab20')
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('Channel index')
    axes[0].set_title('Community assignment over time before temporal consensus')
    axes[0].set_yticks(np.arange(0, len(xyz_loc['formatted_label'].values), 1))
    axes[0].set_yticklabels(xyz_loc['formatted_label'].values)
    fig.colorbar(im1, ax=axes[0])

    # Plot for 'Community assignment over time after temporal consensus'
    im2 = axes[1].imshow(communities_after, aspect='auto', cmap='tab20')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Channel index')
    axes[1].set_title('Community assignment over time after temporal consensus')
    axes[1].set_yticks(np.arange(0, len(xyz_loc['formatted_label'].values), 1))
    axes[1].set_yticklabels(xyz_loc['formatted_label'].values)
    fig.colorbar(im2, ax=axes[1])

    plt.suptitle(f'Community detection using {algorithm} \n')
    plt.tight_layout()
    plt.savefig(output_path+algorithm+'_community_detection.png')
    plt.show()

#### Communities metrics ####
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def jaccard_metric(communities,raw,comparision_set):
    communities_dict=rebuild_communities(communities,raw)  
    
    jaccard_index={}
    for t in range(len(communities_dict)):
        temp_jaccard={}
        for com in communities_dict[t]:
            temp_jaccard[com]=jaccard_similarity(comparision_set,communities_dict[t][com])
        jaccard_index[t]=temp_jaccard

    #Get the maximum for each time step, and the community channels that belong to it 
    max_jaccard=[]
    max_jaccard_com=[]
    for t in range(len(jaccard_index)):
        max_jaccard.append(max(jaccard_index[t].values()))
        max_jaccard_com.append(max(jaccard_index[t], key=jaccard_index[t].get))

    dict_communities={}
    for t in range(len(communities_dict)):
        dict_communities[t]=communities_dict[t][max_jaccard_com[t]]

    return max_jaccard,dict_communities


def jaccard_final_communities(final_communities,comparision_set):
    jaccard_index={}
    for t, com in final_communities.items():
        jaccard_index[t]=jaccard_similarity(comparision_set,com)

    return jaccard_index


#### Final community detection ####



def intra_community_density_with_size_regularization(G, community, alpha=0.6, size_threshold=15):
    """
    Calculate the intra-community density with size regularization for a community in a graph.

    Parameters:
    - G: The graph (networkx.Graph)
    - community: A set of nodes representing the community
    - alpha: Regularization factor to penalize large or small communities (default: 0.1)
    - size_threshold: The ideal size of the community (default: 10)

    Returns:
    - density_score: The intra-community density with regularization
    """
    community = set(community)
    num_nodes = len(community)
    
    if num_nodes <= 3:
        return 0  # If the community is too small, return density as 0

    # Count edges within the community
    intra_edges = G.subgraph(community).number_of_edges()

    # Calculate intra-community density
    possible_edges = num_nodes * (num_nodes - 1) / 2
    if possible_edges == 0:
        density = 0
    else:
        density = intra_edges / possible_edges
    
    # Size regularization: penalize communities that are too small or too large
    size_penalty = alpha * abs(num_nodes - size_threshold) / size_threshold

    # Regularized density score
    density_score = density - size_penalty
    
    return max(0, density_score)  # Ensure non-negative density score



def get_final_communities(communities,raw, tnet):
    """
    Finds the final communities for each time step using the given communities and temporal network.

    Parameters:
    communities (list): List of communities, where each community is represented as a list of nodes.
    raw (mne.Raw object): Raw data object containing channel names.
    tnet (numpy.ndarray): The temporal network data (3D array).
    
    
    """
    communities_dict_after=rebuild_communities(communities,raw)

    max_density_scores=[]
    max_density_scores_com=[]
    best_communities={}
    communities_tuple=[]

    for t in range(len(communities_dict_after)):
        com_t=communities_dict_after[t]
        adj = Adjacency(tnet[:,:,t],labels=raw.ch_names)
        G = nx.Graph(adj.to_graph())
        density_scores = {}
        commun=[]
        for com, ch in com_t.items():
            density_scores[com] = intra_community_density_with_size_regularization(G, ch)
            commun.append(ch)
        max_density_scores.append(max(density_scores.values()))
        #Get the community with the maximum density score
        best_communities[t]=commun[list(density_scores.values()).index(max(density_scores.values()))]
        max_density_scores_com.append(max(density_scores, key=density_scores.get))
        communities_tuple.append((best_communities[t],max_density_scores[t]))

    final_communities={}
    #Get the t and the communities with a density value different from 0
    for t in range(len(best_communities)):
        if max_density_scores[t]!=0:
            final_communities[t]=best_communities[t]

    return final_communities,communities_tuple


def get_max_communities(data_dict):
    final_array = []

    # Iterate through each time step (t)
    for t in range(len(next(iter(data_dict.values())))):  # Assuming all algorithms have the same time steps
        max_density = float('-inf')
        selected_algorithm = None
        selected_community = None

        # Loop through algorithms to find the max density at time t
        for algorithm, communities in data_dict.items():
            community, density = communities[t]

            # Update the best algorithm-community pair for this time step
            if density > max_density:
                max_density = density
                selected_algorithm = algorithm
                selected_community = community

        # Apply the conditional: if the max density is 0, return an empty community
        if max_density == 0:
            selected_community = []

        # Store the result for this time step
        final_array.append((selected_algorithm, [selected_community], max_density))

    return final_array


#### Plotting final metrics ####

def final_metrics_plot(communities_data, inside_networks, output_path,band):
    #Density score
    density_scores = [density for _, _, density in communities_data]
    #Jaccard index
    jaccard_index = []
    for _, community, _ in communities_data:
        jaccard_index.append(jaccard_similarity(inside_networks,community[0]))
    
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].plot(density_scores, label='Density score', color='blue', marker='o')
    ax[0].set_title(f'Max Intra-community density with size regularization over time for {band} band')
    ax[0].set_xlabel('Time step')
    ax[0].set_ylabel('Density score')
    ax[0].legend()

    ax[1].plot(jaccard_index, label='Jaccard index', color='red', marker='o')
    ax[1].set_title(f'Jaccard index over time for {band} band')
    ax[1].set_xlabel('Time step')
    ax[1].set_ylabel('Jaccard index')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(output_path + f'final_metrics_{band}.png')


    
###Threshold calculation###
def plot_weight_distribution_with_threshold(data, outputpath, band, bins=30, percentile=90,method_exp='aec'):
    """
    Plots the density distribution of edge weights over time and adds a magnitude-based threshold line.

    Parameters
    ----------
    data : 3D numpy array
        Temporal network data in the form (nodes, nodes, time).
    bins : int, optional
        Number of bins for the histogram, by default 30.
    percentile : int, optional
        Percentile for magnitude-based threshold, by default 90 (top 10% of weights).
    """

    data=data.transpose(1,2,0)

    num_timepoints = data.shape[2]
    fig, axs = plt.subplots(num_timepoints, 1, figsize=(10, 3 * num_timepoints))
    
    # Check if only one subplot axis is returned
    if num_timepoints == 1:
        axs = [axs]

    thresholds = []

    for t in range(num_timepoints):
        weights = data[:, :, t].flatten()  # Flatten the matrix to get all edge weights
        weights = np.array(weights, dtype=np.float64)  # Convert to numpy array             
        
        # Calculate threshold based on the specified percentile
        threshold_value = np.percentile(weights, percentile)
        thresholds.append(threshold_value)
        # Plot histogram and KDE for the weights
        ax = axs[t]
        sns.histplot(weights, bins=bins, kde=True, ax=ax, color="blue", alpha=0.6)
        
        # Add threshold line to plot
        ax.axvline(threshold_value, color='red', linestyle='--', label=f'Threshold ({percentile}th percentile)')
        
        # Set plot labels, title, and legend
        ax.set_title(f'Time Point {t + 1} - Weight Distribution with Threshold')
        ax.set_xlabel('Edge Weight')
        ax.set_ylabel('Density')
        ax.legend()
    plt.title(f'Edge Weight Distribution with Threshold at {percentile}th Percentile for {band} Band for method {method_exp}')
    plt.tight_layout()
    plt.savefig(outputpath + f'edge_weight_distribution_{band}_{method_exp}.png')
    # plt.show()

    # Calculate the mean of all threshold values
    mean_threshold = np.mean(thresholds)

    # print(f'Selected threshold value at the {percentile}th percentile is: {threshold_value}')
    print(f"Mean threshold value across time points: {mean_threshold}")
    return mean_threshold






    







