import networkx as nx
import numpy as np
import pandas as pd

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

