import networkx as nx
import pandas as pd

# ================= #
#     INDEXING      #
# ================= #

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

# ================= #
#     SELECTION     #
# ================= #

def isolate_LCC(network):
    '''
    Find and return the Largest Connected Component
    in the Human Human Interaction.
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
