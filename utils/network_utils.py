import bct
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
    try:
        LCC = max(conn_comp, key=len)
    except:
        LCC = disease_network.copy()

    return LCC

def get_disease_num_connected_components(interactome_df, disease, from_curated=True):
    '''
    Given the interactome DataFrame,
    Return the number of connected components the disease.
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

    return len(conn_comp)

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


def get_distance_matrix_interactome():

    '''
    Starting from HHI_LCC file, it creates an adjacency matrix considering all the interactome,
    call btcpy library to build the distance matrix and save both in files.
    NB: This distance matrix consider paths which involve also genes of different diseases
    (all the interactome)
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

                #add gene names in the dictionary
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


def get_longest_path_for_a_disease_interactome(disease, predicted_genes = []):

    '''
    Given a disease, get the longest path searching in the rows
    and columns of the distance matrix corresponding to disease genes.
    NB: the paths includes genes not only of this disease beacuse the distance matrix
    is built from the interactome

    - predicted_genes are enriched genes which are added to the disease genes(this parameter is used in disease_analysis.py).
      If call this function after enrichment, give predicted_genes in input
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
    #add predicted genes given in input to disease genes
    disease_genes += predicted_genes

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




def get_longest_path_for_a_disease_LCC(disease, predicted_genes = []):

    '''

    Retrieving LCC genes for the given disease, it computes the adjacency and distance matrix
    for the disease, considering only paths of genes included in the LCC, and returns
    the longest among all paths

    - predicted_genes are enriched genes which are added to the disease genes(this parameter is used in disease_analysis.py)
      If call this function after enrichment, give predicted_genes in input
    '''

    #get lcc genes for the disease
    biogrid = "./data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"
    hhi_df  = select_hhi_only(biogrid)
    lcc_genes = get_disease_LCC(hhi_df, disease)
    #add predicted genes given in input to disease genes
    for elem in predicted_genes:
        lcc_genes.add(elem)


    #create a dictionary with key=gene_name and value=a unique progressive integer to identify the gene
    #(in the dictionary there will be only LCC genes)

    hhi_lcc = './data/HHI_LCC.txt'

    genes_dict = {}
    with open(hhi_lcc, 'r') as of:  #read hh interactions

            i=0
            for line in of:
                #retrieve both gene names
                node1=line.strip().split(',')[0]
                node2=line.strip().split(',')[1]
                if node1 in lcc_genes and node2 in lcc_genes:  #check if both genes are LCC genes
                    #add gene names in the dictionary
                    if (node1 not in genes_dict):
                        genes_dict[node1]= i
                        i+=1
                    if (node2 not in genes_dict):
                        genes_dict[node2]= i
                        i+=1

    #initialize adjacency matrix
    adjacency_matrix = np.zeros((len(lcc_genes), len(lcc_genes)))
    #fill adjacency matrix:
    # (the adjacency matrix is #LCC genes x #LCC genes)
    with open(hhi_lcc, 'r') as of:
        for line in of:
            node1=line.strip().split(',')[0]
            node2=line.strip().split(',')[1]
            #retrieve the id present in the dictionary corresponding to the gene
            if node1 in lcc_genes and node2 in lcc_genes:
                gene1_id = genes_dict[node1]
                gene2_id = genes_dict[node2]
                #set to 1 the cell
                if (gene1_id != gene2_id):
                    adjacency_matrix[gene1_id,gene2_id]=1


    # Save adjacency matrix as binary file
    with open("tmp/adjacency_matrix.npy", "wb") as f:
        np.save(f, adjacency_matrix)

    #call bct library to compute distance matrix (passing adjacency matrix)
    distance_matrix = bct.distance_bin(adjacency_matrix)

    # Save adjacency matrix as binary file
    with open("tmp/distance_matrix.npy", "wb") as f:
        np.save(f, distance_matrix)

    #load distance matrix file
    distance_matrix = './tmp/distance_matrix.npy'
    data = np.load(distance_matrix)

    #loop on the distance matrix: find the max among all the cells
    max=0
    for i in range(len(data)):  #select a row
        for j in range(len(data)):    #select a column
            if i!=j:
                if str(data[i][j]) != 'inf': # 'inf' must be avoided
                    #data[i][j] is the selected cell in the matrix
                    if data[i][j] > max :
                        max = data[i][j]

    return max


def translate_from_stringdb(stringdb_file, stringdb_aliases_file):
    '''
    --> IT WORKS ONLY WITH THESE TWO STRINGDB TXT FILES AS INPUT- MODIFY THE FUNCTION TO USE FOR OTHER PURPOSES.
    
    Given stringdb associations file and aliases file, it translates each gene in the original
    file to the gene symbol and return the dataframe with the associations among genes expressed
    as gene symbols
    '''

    df = pd.read_csv(stringdb_file, sep="\s+", header=0)
    df_aliases = pd.read_csv(stringdb_aliases_file, sep="\t", header=0)

    #create the DataFrame for results
    d = {'protein1': [], 'protein2': []}
    df_translated = pd.DataFrame(data=d)

    #create the dictionary with keys = original StringDB names and values = traslated name
    alias_dict = {}

    j=0
    #TRANSLATION FROM ORIGINAL STRINGDB NAMES TO GENE SYMBOLS
    for i in range(len(df.protein1)): #loop of every row in the associations file
        
        #flags to check if both proteins have an alias
        protein1_exists = False
        protein2_exists = False

        #if protein not in the dictionary
        if df.protein1[i] not in alias_dict:
                
            #find alias in aliases file:
            filter1 = df_aliases['string_protein_id'] == df.protein1[i]  
            filter2 = df_aliases['source'] == 'Ensembl_HGNC'
            aliases_list = df_aliases.loc[filter1 & filter2 ]['alias'].values
            length =len(aliases_list) #df_aliases.loc[filter1 & filter2 ] filters the file, ['alias'] takes the alias column, .values extract the values
            #if you found only one alias
            if length == 1: 
                protein1_exists = True  #the alias has been found
                alias1 = aliases_list[0]   #acces the first and only value
                #set the dictionary
                alias_dict[df.protein1[i]] = alias1

        #if protein has been never traslated  
        else: 
            protein1_exists = True
            #add to the dictionary
            alias1 = alias_dict[df.protein1[i]]


        #let's do again for the second column

        #if protein not in the dictionary
        if df.protein2[i] not in alias_dict:

            #find alias in aliases file:
            filter1 = df_aliases['string_protein_id'] == df.protein2[i]  
            filter2 = df_aliases['source'] == 'Ensembl_HGNC'
            aliases_list = df_aliases.loc[filter1 & filter2 ]['alias'].values
            length = len(aliases_list) #df_aliases.loc[filter1 & filter2 ] filters the file, ['alias'] takes the alias column, .values extract the values
            #if you found only one alias
            if length == 1:
                protein2_exists = True     #the alias has been found         
                alias2 = aliases_list[0]   #acces the first and only value
                #set the dictionary
                alias_dict[df.protein2[i]] = alias2
        #if protein has been never traslated          
        else:
            protein2_exists = True
            #add to the dictionary
            alias2 = alias_dict[df.protein2[i]]

        #if both proteins has been traslated we can add a row in the dataframe
        if protein1_exists and protein2_exists:
            df_translated.loc[j] = [alias1, alias2]
            j+=1

    return df_translated


# =================================
#     B U I L D  N E T W O R K S
# =================================

def build_network_from_biogrid(biogrid_database, hhi_only=False, physical_only=False, remove_self_loops=True):
    """
    Given the path for a BIOGRID protein-protein interaction database,
    build the graph.
    """

    # Read the database and build the currespondent DataFrame
    df = pd.read_csv(biogrid_database, sep="\t", header=0)

    # Select only human-human interactions
    if hhi_only == True:
        df = df.loc[(df["Organism ID Interactor A"] == 9606) &
                    (df["Organism ID Interactor B"] == 9606)]

    # Select only physical interactions
    if physical_only == True:
        df = df.loc[df["Experimental System Type"] == "physical"]

    # Build the graph
    G = nx.from_pandas_edgelist(df,
                                source = "Official Symbol Interactor A",
                                target = "Official Symbol Interactor B",
                                create_using=nx.Graph())  #nx.Graph doesn't allow duplicated edges

    # Remove self loops
    if remove_self_loops == True:
        self_loop_edges = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loop_edges)

    return G


def build_network_from_stringdb(stringdb_database, remove_self_loops=True):
    """
    Given the path for a StringDB protein-protein interaction database,
    build the graph.
    """

    # Read the database and build the currespondent DataFrame
    df = pd.read_csv(stringdb_database, sep="\s+", header=0)

    # Build the graph
    G = nx.from_pandas_edgelist(df,
                                source = "protein1",
                                target = "protein2",
                                create_using=nx.Graph())  #x.Graph doesn't allow duplicated edges

    # Remove self loops
    if remove_self_loops == True:
        self_loop_edges = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loop_edges)

    return G

def build_network_from_pnas(stringdb_database, remove_self_loops=True):
    """
    Given the path for the PNAS protein-protein interaction database,
    build the graph.
    """

    # Read the database and build the currespondent DataFrame
    df = pd.read_csv(stringdb_database, header=0)

    # Build the graph
    G = nx.from_pandas_edgelist(df,
                                source = "proteinA_entrezid",
                                target = "proteinB_entrezid",
                                create_using=nx.Graph())  #x.Graph doesn't allow duplicated edges

    # Remove self loops
    if remove_self_loops == True:
        self_loop_edges = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loop_edges)

    return G

def build_network_from_diamond_dataset(diamond_interactome, remove_self_loops=True):
    """
    Given the path for the interactome used by DIAMOnD authors,
    build the graph.
    """

    # Read the tsv file
    df = pd.read_csv(diamond_interactome, delimiter="\t", header=0)

    # Build the graph
    G = nx.from_pandas_edgelist(df,
                                source = "gene_ID_1",
                                target = "gene_ID_2",
                                create_using=nx.Graph())  #x.Graph doesn't allow duplicated edges

    # Remove self loops
    if remove_self_loops == True:
        self_loop_edges = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loop_edges)

    return G


def LCC(G):
    '''
    Given a graph G, find and return its Largest Connected Component.
    '''

    # Find the connected components
    conn_comp = list(nx.connected_components(G))
    print(f"# of connected components: {len(conn_comp)}")

    # Sort the connected components by descending size
    conn_comp_len = [len(c) for c in sorted(conn_comp, key=len,
                                        reverse=True)]
    print(f"Lengths of connected components: {conn_comp_len}")

    # isolate the LCC
    LCC = max(conn_comp, key=len)
    print(f"LCC len: {len(LCC)}")

    LCC_hhi = G.subgraph(LCC).copy()

    return LCC_hhi
