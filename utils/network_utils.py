from pyexpat.model import XML_CQUANT_NONE
import sys
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import bct 


from utils.data_utils import get_disease_genes_from_gda

# ========================
#     I N D E X I N G
# ========================

def node_to_index(network, nodes_of_interest):
    '''
    Given a network and a list of nodes in that network
    retrive the indices of those nodes.
    '''

    all_nodes = list(network.nodes)
    indices = []

    for idx, node in enumerate(all_nodes):
        if node in nodes_of_interest:
            indices.append(idx)

    return indices

def index_to_node(network, indices):
    '''
    Given a network and a list of nodes in that network
    retrive the indices of those nodes.
    '''

    all_nodes = list(network.nodes)
    nodes_of_interest = []

    for idx, node in enumerate(all_nodes):
        if idx in indices:
            nodes_of_interest.append(node)

    return nodes_of_interest

# ============
# I N / O U T
# ============

def write_edges_in_file(network, output_file):
    '''
    Print all the connections in a network a file
    with the following schema:
        Node1,Node2
        Node3,Node4
        ...,...
    '''
    with open(output_file, 'w') as fo:
        LCC_hhi_edges = list(network.edges)
        for e in LCC_hhi_edges:
            fo.write(e[0] + "," + e[1] + "\n")

# ===================
#  S E L E C T I O N
# ===================

def isolate_LCC(network):
    '''
    Find and return the Largest Connected Component
    of the network.
    '''

    # find connected components
    conn_comp = list(nx.connected_components(network))
    print(f"# of connected components: {len(conn_comp)}")
    # print(conn_comp)

    # sort the connected components by descending size
    conn_comp_len = [len(c) for c in sorted(conn_comp, key=len,
                                        reverse=True)]
    print(f"Lengths of connected components: {conn_comp_len}")

    # isolate the LCC
    LCC = max(conn_comp, key=len)
    print(f"LCC len: {len(LCC)}")

    LCC_hhi = network.subgraph(LCC).copy()

    return LCC_hhi

def select_hhi_only(filename, only_physical=1):
    '''
    Select from the BIOGRID only the Human-Human Interactions.
    If only_physical == 1, remove all non physical interactions.
    Return a DataFrame with the filtered connections.
    '''

    # select HHIs
    df = pd.read_csv(filename, sep="\t", header=0)
    df = df.loc[(df["Organism ID Interactor A"] == 9606) &
                (df["Organism ID Interactor B"] == 9606)]

    # select only physical interactions
    if only_physical == 1:
        df = df.loc[df["Experimental System Type"] == "physical"]

    return df

def select_disease_interactions_only(hhi_df, disease, curated=True):
    '''
    From the Human-Human interactions select only the interactions regarding
    the disease.
    '''

    # get disease genes from GDS
    if curated:
        gda_filename = "data/curated_gene_disease_associations.tsv"
    else:
        gda_filename = "data/all_gene_disease_associations.tsv"

    disease_genes = get_disease_genes_from_gda(gda_filename, disease)

    # convert to an immutable object
    disease_genes_array = np.array(disease_genes)

    # select from human-human interactions only the rows that
    # have both a gene belonging to disease_genes
    disease_df = hhi_df.loc[(hhi_df["Official Symbol Interactor A"].isin(disease_genes_array)) &
                            (hhi_df["Official Symbol Interactor B"].isin(disease_genes_array))]

    return disease_df

def get_disease_LCC(interactome_df, disease, from_curated=True):
    '''
    Given the interactome DataFrame,
    Return the LCC of the disease.
    '''

    # From the dataframe select the disease genes only
    disease_df = select_disease_interactions_only(interactome_df, disease, curated=from_curated)

    # Create the network
    disease_network = nx.from_pandas_edgelist(disease_df,
                                              source = "Official Symbol Interactor A",
                                              target = "Official Symbol Interactor B",
                                              create_using=nx.Graph())  #x.Graph doesn't allow duplicated edges
    # Remove self loops
    self_loop_edges = list(nx.selfloop_edges(disease_network))
    disease_network.remove_edges_from(self_loop_edges)

    # Find connected components
    conn_comp = list(nx.connected_components(disease_network))

    # Isolate the LCC
    try:
        LCC = max(conn_comp, key=len)
    except:
        LCC = disease_network.copy()

    return LCC

def get_genes_percentage(seed_genes, LCC):
    '''
    Given the seed genes and the LCC of a network,
    compute the ratio between the LCC size and the
    total number of seed genes.
    '''
    return nx.number_of_nodes(LCC)/len(seed_genes)

def get_density(G):

    '''Computes the density of a given graph'''
    d = nx.density(G)

    return d


def get_distance_matrix():

    '''
    Starting from HHI_LCC file, it creates an adjacency matrix,
    call btcpy library to build the distance matrix and same both 
    in files
    '''

    hhi_lcc = './data/HHI_LCC.txt'

    genes_dict = {}
  
    #create a dictionary with key=gene_name and value=a unique progressive integer to identify the gene
    with open(hhi_lcc, 'r') as of:  #read hh interactions

            i=0
            for line in of:
                #retrieve both gene names
                node1=line.strip().split(',')[0]
                node2=line.strip().split(',')[1]
                
                #add gene names in te dictionary
                if (node1 not in genes_dict):
                    genes_dict[node1]= i
                    i+=1
                if (node2 not in genes_dict):
                    genes_dict[node2]= i
                    i+=1

    #create an adjacency matrix filled of zeros
    adjacency_matrix = np.zeros((len(genes_dict), len(genes_dict)))


    #fill adjacency matrix with 1 where it needs
    with open(hhi_lcc, 'r') as of:
        
        for line in of:
            node1=line.strip().split(',')[0]
            node2=line.strip().split(',')[1]
            #retrieve the id present in the dictionary corresponding to the gene
            gene1_id = genes_dict[node1]
            gene2_id = genes_dict[node2]

            #set to 1 the cell
            if (gene1_id != gene2_id):
                adjacency_matrix[gene1_id,gene2_id]=1

    #IT'S COMMENTED TO AVOID TO OVERWRITE

    ## Save adjacency matrix as binary file
    #with open("data/adjacency_matrix.npy", "wb") as f:
    #    np.save(f, adjacency_matrix)
    #
    #
    ##call bct library to compute distance matrix (passing adjacency matrix)
    #distance_matrix = bct.distance_bin(adjacency_matrix)
    #
    ## Save adjacency matrix as binary file
    #with open("data/distance_matrix.npy", "wb") as f:
    #    np.save(f, distance_matrix)
    
    return 


def get_longest_path_for_a_disease(disease):

    '''
    Given a disease, get the longest path searching in the rows
    and columns of the distance matrix corresponding to disease genes
    '''

    hhi_lcc = './data/HHI_LCC.txt'
    genes_dict = {}
  
    #create a dictionary with key=gene_name and value=a unique progressive integer to identify the gene
    with open(hhi_lcc, 'r') as of:  #read hh interactions

            i=0
            for line in of:
                #retrieve both gene names
                node1=line.strip().split(',')[0]
                node2=line.strip().split(',')[1]
                
                #add gene names in te dictionary
                if (node1 not in genes_dict):
                    genes_dict[node1]= i
                    i+=1
                if (node2 not in genes_dict):
                    genes_dict[node2]= i
                    i+=1
    
    #load distance matrix file
    distance_matrix = './data/distance_matrix.npy'
    data = np.load(distance_matrix)
    
    #get genes for this disease
    gda_filename = "./data/curated_gene_disease_associations.tsv"
    disease_genes = get_disease_genes_from_gda(gda_filename, disease)

    #for each gene of the disease, retrieve its id in the genes dictionary and add to ids
    ids = []
    for gene in disease_genes:
        if gene in genes_dict:   #not all the genes are in the dictionary (there aren't those ones which don't have interactions)
            gene_id = genes_dict[gene] 
            ids.append(gene_id)

    #loop on the distance matrix (considering only rows and columns corresponding to disease genes)    
    max=0
    for id in ids:  #select a row
        for column in ids:    #select a column
            if id!=column :
                if str(data[id][column]) != 'inf': # 'inf' must be avoided
                    #data[id][column] is the selected cell in the matrix
                    if data[id][column] > max :
                        max = data[id][column]
    return max
