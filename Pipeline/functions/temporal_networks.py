import numpy as np
from nltools.data import Adjacency
import networkx as nx
import pandas as pd
from functools import reduce
import teneto
from teneto import plot 
from teneto import TemporalNetwork
import matplotlib.pyplot as plt




#### Graph metrics ####


### Temporal plots ###

def plot_temporal_graph(tnet, output_path, xyz_loc):
    fig,ax = plt.subplots(figsize=(25,15))
    ax = plot.slice_plot(tnet.network, ax, plotedgeweights=False,cmap='Pastel2', nodelabels=list(xyz_loc['formatted_label'].values))
    plt.tight_layout()


### Community detection ###
def detect_communities(data, xyz_loc, raw, output_path, threshold_level=0.15, algorithm='k_clique_communities', k=2, plot=False):
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
    tnet_bu.binarize(threshold_type='percent', threshold_level=threshold_level)
    
    # Get the binarized adjacency network
    tnet_bu_ar = tnet_bu.network
    
    # Initialize a dictionary to store communities
    communities_dict = {}
    
    # Detect communities at each time step
    for i in range(tnet_bu_ar.shape[2]):
        # Create a network from the adjacency matrix
        adj = Adjacency(tnet_bu_ar[:, :, i], labels=raw.ch_names)
        G = nx.Graph(adj.to_graph())
        
        # Select the community detection algorithm
        communities_dict[i] = communities_algorithm(G, algorithm, k)
    
    # Initialize the output array to store community assignments before consensus
    num_timesteps = len(communities_dict)
    num_channels = len(raw.ch_names)
    output = [[-1 for _ in range(num_timesteps)] for _ in range(num_channels)]
    
    # Assign communities to the channels
    for t, subcommunities in communities_dict.items():
        for subcom_idx, subcommunity in enumerate(subcommunities):
            for channel in subcommunity:
                if channel in raw.ch_names:
                    channel_idx = raw.ch_names.index(channel)
                    output[channel_idx][t] = subcom_idx
    
    # Perform temporal consensus
    communities_after = teneto.communitydetection.make_temporal_consensus(output)

    # Plot the community assignment over time
    if plot:
        plot_community_assignment(communities_after,output,algorithm,output_path)
    
    print(f'Community detection using {algorithm} completed.')
    
    return output, communities_after


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
        communities_generator = list(nx.community.k_clique_communities(G, k))
        communities = [list(c) for c in communities_generator]
    
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


def plot_community_assignment(communities_after,output,algorithm,output_path,xyz_loc):
    #Compare the communities
    fig, axes = plt.subplots(1, 2, figsize=(15, 20))

    # Plot for 'Community assignment over time before temporal consensus'
    im1 = axes[0].imshow(output, aspect='auto', cmap='tab20')
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
    communities_dict_after={}
    for t in range(len(communities[0])):
        temp_com={}
        for i, ch in enumerate(raw.ch_names):
            temp_com[communities[i][t]]=temp_com.get(communities[i][t],[])+[ch]
        communities_dict_after[t]=temp_com     
    
    jaccard_index={}
    for t in range(len(communities_dict_after)):
        temp_jaccard={}
        for com in communities_dict_after[t]:
            temp_jaccard[com]=jaccard_similarity(comparision_set,communities_dict_after[t][com])
        jaccard_index[t]=temp_jaccard

    #Get the maximum for each time step, and the community channels that belong to it 
    max_jaccard=[]
    max_jaccard_com=[]
    for t in range(len(jaccard_index)):
        max_jaccard.append(max(jaccard_index[t].values()))
        max_jaccard_com.append(max(jaccard_index[t], key=jaccard_index[t].get))

    return max_jaccard,max_jaccard_com


    
    







