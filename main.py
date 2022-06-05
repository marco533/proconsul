# =====================
#     I M P O R T S
# =====================

import argparse
import sys
from test.extended_validation import *
from test.kfold_validation import *

import networkx as nx
import numpy as np
import pandas as pd

from utils.network_utils import *
from utils.data_utils import *




# ===========================
#     R E A D   I N P U T
# ===========================

def print_usage():
    print(' ')
    print('        usage: python3 project.py --algs --validation --disease_file --database --heat_diffusion_time --pdiamond_n_iters --pdiamond_temp --pdiamond_top_p --pdiamond_top_k')
    print('        -----------------------------------------------------------------')
    print('        algs                     : List of algorithms to use to collect results.')
    print('                                   They can be: "diamond", "diamond2, "pdiamond", "pdiamond_log", "pdiamond_entropy", "heat_diffusion"')
    print('                                   If all, run all the algorithms. (default: all')
    print('        validation               : Type of validation on which test the algorithms. It can be')
    print('                                   "kfold", "extended" or "all".')
    print('                                    If all, perform both the validations. (default: all')
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison')
    print('                                   (default: "data/disease_file.txt).')
    print('        database                 : Database name from which take the PPIs. Choose from "biogrid" or "stringdb".')
    print('                                   (default: "biogrid)')
    print('        heat_diffusion_time      : Diffusion time for Heat Diffusion algorithm.')
    print('                                   (default: 0.005)')
    print('        pdiamond_n_iters         : Number of iteration for pDIAMOnD.')
    print('                                   (default: 10)')
    print('        pdiamond_temp            : Temperature value for the pDIAMOnD softmax function.')
    print('                                   (default: 1.0)')
    print('        pdiamond_top_p           : Probability threshold value for pDIAMOnD nucleus sampling. If 0 no nucleus sampling')
    print('                                   (default: 0.0)')
    print('        pdiamond_top_k           : Length of the pvalues subset for Top-K sampling. If 0 no top-k sampling')
    print('                                   (default: 0)')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set disease, algorithms and validation')
    parser.add_argument('-a','--algs', nargs='+', default=["diamond", "pdiamond", "pdiamond_log", "pdiamond_entropy" "heat_diffusion"],
                    help='List of algorithms to run (default: all)')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation. (default: all')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Relative path to the file with disease names (default: "data/disease_file.txt)')
    parser.add_argument('--database', type=str, default="biogrid",
                    help='Database name (default: "biogrid')
    parser.add_argument('--heat_diffusion_time', type=float, default=0.005,
                    help='Diffusion time for heat_diffusion algorithm. (default: 0.005')
    parser.add_argument('--pdiamond_n_iters', type=int, default=10,
                    help='Number of iteration for pDIAMOnD. (default: 10)')
    parser.add_argument('--pdiamond_temp', type=float, default=1.0,
                    help='Temperature value for the pDIAMOnD softmax function. (default: 1.0)')
    parser.add_argument('--pdiamond_top_p', type=float, default=0.0,
                    help='Probability threshold value for pDIAMOnD nucleus sampling. (default: 0.0)')
    parser.add_argument('--pdiamond_top_k', type=int, default=0,
                    help='Length of the pvalues subset for Top-K sampling. (default: 0)')
    return parser.parse_args()

def read_terminal_input(args):
    '''
    Read the arguments passed by command line.
    '''

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

    # Read parsed values
    algs                = args.algs
    validation          = args.validation
    disease_file        = args.disease_file
    database            = args.database
    heat_diffusion_time = args.heat_diffusion_time
    pdiamond_n_iters    = args.pdiamond_n_iters
    pdiamond_temp       = args.pdiamond_temp
    pdiamond_top_p      = args.pdiamond_top_p
    pdiamond_top_k      = args.pdiamond_top_k

    # 1. Check algorithm names
    for alg in algs:
        if alg not in ["diamond", "diamond2", "pdiamond", "pdiamond_log", "pdiamond_entropy", "heat_diffusion"]:
            print(f"ERROR: {alg} is not a valid algorithm!")
            print_usage()
            sys.exit(0)

    # 2. Check validation
    if validation not in ["kfold", "extended", "all"]:
        print(f"ERROR: {validation} is no valid validation method!")
        print_usage()
        sys.exit(1)

    # Get list of validations
    if validation == 'all':
        validations = ['kfold', 'extended']
    else:
        validations = [validation]

    # 3. Get diesease list from file
    try:
        diseases = read_disease_file(disease_file)
    except:
        print(f"Not found file in {disease_file} or no valid location.")
        print_usage()
        sys.exit(1)

    # if empty, exit with error
    if len(diseases) == 0:
        print(f"ERROR: No diseases in disease_file")
        sys.exit(1)

    # 4. Check database
    if database not in ["biogrid", "stringdb"]:
        print("ERROR: no valid database name")
        print_usage()
        sys.exit(0)

    if database == "biogrid":
        database_path = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"

    if database == "stringdb":
        database_path = "data/9606.protein.links.full.v11.5.txt"

    # 5. Check diffusion time
    if heat_diffusion_time < 0:
        print(f"ERROR: diffusion time must be greater than 0")
        print_usage()
        sys.exit(1)

    # 6. Check pDIAMOnD number of iterations
    if pdiamond_n_iters <= 0:
        print(f"ERROR: pdiamond_n_iters must be greater or equal 1")
        print_usage()
        sys.exit(1)

    # 7. Check pDIAMOnD temperature
    if pdiamond_temp < 0:
        print(f"ERROR: pdiamond_temp must be greater or equal 0")
        print_usage()
        sys.exit(1)

    # 8. Check pDIAMOnD top-p sampling
    if pdiamond_top_p < 0 or pdiamond_top_p > 1:
        print("ERROR: top_p must be in [0,1]")
        print_usage()
        sys.exit(1)

    # 9. Check pDIAMOnD top-k sampling
    if pdiamond_top_k < 0:
        print("ERROR: pdiamond_top_k must be greater or equal 0")
        print_usage()
        sys.exit(1)

    print('                                                    ')
    print(f"===================================================")
    print(f"Algorithms: {algs}"                                 )
    print(f"Validations: {validations}"                         )
    print(f"Diseases: {len(diseases)}"                          )
    print(f"Database: {database}"                               )
    print(f"Heat Diffusion Time: {heat_diffusion_time}"         )
    print(f"pDIAMOnD number of iterations: {pdiamond_n_iters}"  )
    print(f"pDIAMOnD temperature: {pdiamond_temp}"  )
    print(f"pDIAMOnD top-p: {pdiamond_top_p}"                   )
    print(f"pDIAMOnD top-k: {pdiamond_top_k}"                   )
    print(f"===================================================")
    print('                                                    ')


    return algs, validations, diseases, database, database_path, heat_diffusion_time, pdiamond_n_iters, pdiamond_temp, pdiamond_top_p, pdiamond_top_k



# =========================
#     U T I L I T I E S
# =========================

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


# main
if __name__ == "__main__":

    # Read input
    args = parse_args()
    algs, validations, diseases, database_name, database_path, heat_diffusion_time, pdiamond_n_iters, pdiamond_temp, pdiamond_top_p, pdiamond_top_k = read_terminal_input(args)

    # Compact all the algorithm hyperparameters in a dictionary
    hyperparams = {"heat_diffusion_time": heat_diffusion_time,
                   "pdiamond_n_iters": pdiamond_n_iters,
                   "pdiamond_temp": pdiamond_temp,
                   "pdiamond_top_p": pdiamond_top_p,
                   "pdiamond_top_k": pdiamond_top_k}

    # Build the Human-Human Interactome
    if database_name == "biogrid":
        hhi = build_network_from_biogrid(database_path,
                                        hhi_only=True,
                                        physical_only=True,
                                        remove_self_loops=True)
    if database_name == "stringdb":
        hhi = build_network_from_stringdb(database_path,
                                          remove_self_loops=True)

    # Isolate the Largest Connected Component
    hhi_lcc = LCC(hhi)

    # -------------------------------
    #     K-FOLD CROSS VALIDATION
    # -------------------------------

    gda_curated = "data/curated_gene_disease_associations.tsv"

    if 'kfold' in validations:
        for alg in algs:
            for disease in diseases:

                disease_genes = get_disease_genes_from_gda(gda_curated, disease, translate_in=database_name)

                # check that the list of disease genes is not empty
                if len(disease_genes) == 0:
                    print(f"WARNING: {disease} has no disease genes. Skip this disease")
                    continue

                k_fold_cross_validation(hhi_lcc,
                                        alg,
                                        disease,
                                        disease_genes,
                                        K=5,
                                        database_name=database_name,
                                        hyperparams=hyperparams)


    # --------------------------------
    #       EXTENDED VALIDATION
    # --------------------------------

    gda_all = "data/all_gene_disease_associations.tsv"

    if 'extended' in validations:
        for alg in algs:
            for disease in diseases:

                # get disease genes from curated and all GDA
                curated_disease_genes = get_disease_genes_from_gda(gda_curated, disease, translate_in=database_name)
                all_disease_genes = get_disease_genes_from_gda(gda_all, disease, translate_in=database_name)

                # remove from all the genes that are already in curated
                extended_genes = list(set(all_disease_genes) - set(curated_disease_genes))

                # check if the disease genes lists are not empty
                if len(curated_disease_genes) == 0:
                    print(f"WARNING: {disease} has no curated disease genes. Skip this disease")
                    continue

                if len(all_disease_genes) == 0:
                    print(f"WARNING: {disease} has no all disease genes. Skip this disease")
                    continue

                if len(extended_genes) == 0:
                    print(f"WARNING: {disease} has no extended disease genes. Skip this disease")
                    continue

                # run the extended validation on {algorithm}
                extended_validation(hhi_lcc,
                                    alg,
                                    disease,
                                    curated_disease_genes,
                                    extended_genes,
                                    database_name=database_name,
                                    hyperparams=hyperparams)
