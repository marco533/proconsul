import argparse
import csv
import sys
from itertools import combinations
from test.kfold_validation import *

import seaborn as sns
from matplotlib import pyplot as plt
from numpy import indices

from utils.data_utils import get_disease_genes_from_gda, string_to_filename
from utils.network_utils import *
from utils.network_utils import (get_density, get_disease_LCC,
                                 get_longest_path_for_a_disease_interactome,
                                 get_longest_path_for_a_disease_LCC)

# =======================
#   R E A D   I N P U T
# =======================

def print_usage():
    print(' ')
    print('        usage: python3 pdiamond_hyperparams_tuning.py --metrics --validations --n_iterations --disease_file')
    print('        -----------------------------------------------------------------')
    print('        metrics                  : Metrics used for tuning. They can be "precision", "recall", "f1" and "ndcg"')
    print('                                   (default: all')
    print('        validations              : Type of validation on which tune the hyperparameters.')
    print('                                   They can be "kfold" or "extended" (default: all).')
    print('        disease_file             : Relative path to the file containing the disease names to use for the tuning')
    print('                                   (default: "data/disease_file.txt).')
    print('        n_iteration              : Number of iteration for pDIAMOnD.')
    print('                                   (default: 10)')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set values for hyperparameters tuning.')
    parser.add_argument('--metrics', nargs='+', default=["precision", "recall", "f1", "ndcg"],
                    help='Metrics for tuning. (default: all')
    parser.add_argument('--validations', nargs='+', default=["kfold", "extended"],
                    help='Type of validation on which tune the hyperparameters. (default: all')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Relative path to the file containing the disease names to use for the comparison (default: "data/disease_file.txt)')
    parser.add_argument('--n_iterations', type=int, default=10,
                    help='Number of iteration of pDIAMOnD. (default: 10)')
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
                if line[0] == "#":  # Skip commented lines
                    continue
                disease_list.append(line.replace("\n",""))

        return disease_list

    # Read the parsed values.
    metrics         = args.metrics
    validations     = args.validations
    disease_file    = args.disease_file
    n_iterations    = args.n_iterations


    # Get disease list from file.
    try:
        disease_list = read_disease_file(disease_file)
    except:
        print(f"Not found file in {disease_file} or no valid location.")
        sys.exit(1)

    # Check disease list.
    if len(disease_list) == 0:
        print(f"ERROR: No diseases in disease_file.")
        sys.exit(1)

    # Check validations.
    for val in validations:
        if val not in ["kfold", "extended"]:
            print(f"ERROR: {val} is no an correct validation.")
            print_usage()
            sys.exit(1)

    # Check number of pdiamond iterations.
    if n_iterations <= 0:
        print(f"ERROR: n_iteration must be greater or equal of 1")
        sys.exit(0)

    print('                                                            ')
    print(f"-----------------------------------------------------------")
    print(f"Metrics: {metrics}"                                         )
    print(f"Validations: {validations}"                                 )
    print(f"Diseases: {len(disease_list)}"                              )
    print(f"Number of iterations: {n_iterations}"                       )
    print(f"-----------------------------------------------------------")
    print('                                                            ')

    return metrics, validations, disease_list, n_iterations





# =========================
#     U T I L I T I E S
# =========================

def build_network_from_database(database, hhi_only=False, physical_only=False, remove_self_loops=True):
    """
    Given the path for a protein-protein interaction database,
    build the graph.
    """

    # Read the database and build the currespondent DataFrame
    df = pd.read_csv(database, sep="\t", header=0)

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




# ===============
#     M A I N
# ===============

if __name__ == "__main__":

    # Read input
    args = parse_args()
    metrics, validations, diseases, n_iterations = read_terminal_input(args)

    # Build the Human-Human Interactome
    database = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"
    hhi  = build_network_from_database(database,
                                       hhi_only=True,
                                       physical_only=True,
                                       remove_self_loops=True)

    # Isolate the Largest Connected Component
    hhi_lcc = LCC(hhi)

    # Hyperparameters tuning
    hyperparameters = ["top_k", "top_p", "starting_temp", "temp_increase_step", "temp_increase_epoch"]

    top_p = 1.0
    top_k = 0
    starting_temp = 0.1
    temp_increase_step = 0.1
    temp_increase_epoch = 0

    curated_gda = "data/curated_gene_disease_associations.tsv"
    extended_gda = "data/all_gene_disease_associations.tsv"

    for disease in diseases:

        # get disease genes from curated GDA
        disease_genes = get_disease_genes_from_gda(curated_gda, disease)

        # run the k-fold validation on {algorithm}
        k_fold_cross_validation(hhi_lcc,
                                disease_genes,
                                "pdiamond_complete",
                                disease,
                                K=5,
                                num_iters_pdiamond=n_iterations)


