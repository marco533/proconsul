import networkx as nx
import numpy as np
import pandas as pd
from itertools import combinations

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
    LCC = max(conn_comp, key=len)

    return LCC

def get_genes_percentage(seed_genes, LCC):
    '''
    Given the seed genes and the LCC of a network,
    compute the ratio between the LCC size and the
    total number of seed genes.
    '''
    return len(LCC)/len(seed_genes)

def get_density(G):

    '''Computes the density of a given graph'''
    d = nx.density(G)

    return d 

def get_longest_paths():

    def read_disease_file(disease_file):
        '''
        Read the disease file and return a list of diseases.
        The file MUST HAVE only a desease name for each line.
        '''
        disease_list = []
        with open(disease_file, 'r') as df:
            for line in df:
                if line[0] == "#":  # skip commented lines
                    continue
                disease_list.append(line.replace("\n",""))
        return disease_list

    disease_file = 'data/disease_file.txt'

    # get disease list from file
    try:
        disease_list = read_disease_file(disease_file)
    except:
        print(f"Not found file in {disease_file} or no valid location.")
        sys.exit(0)

    # if list is empty fill it with default diseases
    if len(disease_list) == 0:
        print(f"ERROR: No diseases in disease_file")
        sys.exit(0)

    #create network:        

    # select the human-human interactions from biogrid
    biogrid_file = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"
    hhi_df = select_hhi_only(biogrid_file)

    # create the hhi from the filtered data frame
    hhi = nx.from_pandas_edgelist(hhi_df,
                                  source = "Official Symbol Interactor A",
                                  target = "Official Symbol Interactor B",
                                  create_using=nx.Graph())  #x.Graph doesn't allow duplicated edges
    print("Network Info:")
    print(nx.info(hhi), end="\n\n")

    # remove self loops
    self_loop_edges = list(nx.selfloop_edges(hhi))
    hhi.remove_edges_from(self_loop_edges)
    print("After removing self-loops:")
    print(nx.info(hhi), end="\n\n")

    # isolate the largest connected component
    LCC_hhi = isolate_LCC(hhi)

    print("Isolating the LCC:")
    print(nx.info(LCC_hhi), end="\n\n")


    curated_gda_filename = "data/curated_gene_disease_associations.tsv"
    all_gda_filename = "data/all_gene_disease_associations.tsv"

    results = {}
    #there are 2 modes: search only on LCC or not only in LCC
    for mode in ['LCC', 'not_only_LCC']:

        for disease in disease_list:

            #LCC mode:

            if mode == 'LCC':
                #in LCC mode selected_genes are only LCC genes in hhi
                selected_genes = get_disease_LCC(hhi_df, disease, from_curated=True)
                #print(f'genes in LCC: {len(selected_genes)}')

                #build a copy of the graph dropping not-LCC genes
                hhi_copy = hhi.copy()
                for gene in hhi.nodes():
                    if gene not in selected_genes:
                        hhi_copy.remove_node(gene)

                #print(len(hhi_copy.nodes()))
                max=0
                #compute all possible pairs among LCC nodes
                pairs = list(combinations(hhi_copy.nodes(), 2))
                #find out maximum path among all possible ones
                for i in range(len(pairs)):
                    #all_simple_paths takes the graph and two nodes and return all possible paths
                    for path in nx.all_simple_paths(hhi_copy, pairs[i][0], pairs[i][1]):
                        if (len(path) > max):
                            max = len(path)
                print(f'{disease}, {max}')
                results[disease]=max

    return results

            #not only LCC:

#            if mode == 'not_only_LCC':
#                # get disease genes from curated GDA
#                curated_disease_genes = get_disease_genes_from_gda(curated_gda_filename, disease)
#           
#                #In not only LCC mode, selected_genes are all the curated disease-genes
#                selected_genes = curated_disease_genes
#                #print(f'genes not only in LCC: {len(selected_genes)}')
#
#                max=0
#                #compute all possible pairs
#                pairs = list(combinations(selected_genes, 2))
#                #print(len(pairs))
#                #find out maximum path among all possible ones
#                for i in range(len(pairs)):
#                    #all_simple_paths takes the graph and two nodes and return all possible paths
#                    for path in nx.all_simple_paths(hhi, pairs[i][0], pairs[i][1]):
#                        if (len(path) > max):
#                            max = len(path)
#                print(f'{disease}, {max}')
#                results[disease]=max    



  
    
