'''
Heat Diffusion based on:
https://github.com/idekerlab/heat-diffusion/blob/c434a01913c3a35e2c189955f135f050be246379/heat_diffusion_service.py#L39
'''

import operator
import networkx
import copy

from numpy import array
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
from utils.network_utils import node_to_index, index_to_node

def diffuse(matrix, heat_array, time):
    return expm_multiply(-matrix, heat_array, start=0, stop=time, endpoint=True)[-1]

def create_sparse_matrix(network, normalize=False):
    if normalize:
        return csc_matrix(networkx.normalized_laplacian_matrix(network))
    else:
        return csc_matrix(networkx.laplacian_matrix(network))

def create_heat_array(network, seed_genes, heat_value=1.0):
    heat_list = []
    for node in network.nodes:
        if node in seed_genes:
            heat_list.append(heat_value)
        else:
            heat_list.append(0.0)

    return array(heat_list)

def filter_node_list(node_list, nodes_to_remove):
    filtered_nodes = []
    for item in node_list:
        if item[0] not in nodes_to_remove:
            filtered_nodes.append(item[0])

    return filtered_nodes

def run_heat_diffusion(network, seed_genes, diffusion_time=0.005, n_positions=100):
    # get sparse matrix representation of the network
    matrix = create_sparse_matrix(network, normalize=True)

    # create heat array
    heat_array = create_heat_array(network, seed_genes)

    # heat_diffusion
    diffused_heat_array = diffuse(matrix, heat_array, diffusion_time)

    # get the heat of each node
    node_heat = {node_name: diffused_heat_array[i] for i, node_name in enumerate(network.nodes())}

    # sort the nodes by descending order
    sorted_nodes = sorted(node_heat.items(), key=lambda x:x[1], reverse=True)
    # print(len(sorted_nodes))

    # remove from the sorted nodes the seed genes
    predicted_genes = filter_node_list(sorted_nodes, seed_genes)
    # print(len(predicted_genes))

    return predicted_genes[:n_positions]

