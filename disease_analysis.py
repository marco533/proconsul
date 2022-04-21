import argparse
import csv

import numpy as np
import pandas as pd

from utils.network_utils import *

# =======================
#   R E A D   I N P U T
# =======================

def print_usage():
    print(' ')
    print('        usage: python3 disease_analysis.py --disease_file ')
    print('        -----------------------------------------------------------------')
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison')
    print('                                   (default: "data/disease_file.txt).')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Get algorithms to compare and on which validation.')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Relative path to the file containing the disease names to use for the comparison (default: "data/disease_file.txt)')

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

    # Read the parsed values
    disease_file    = args.disease_file

    # Get disease list from file
    try:
        disease_list = read_disease_file(disease_file)
    except:
        print(f"Not found file in {disease_file} or no valid location.")
        sys.exit(0)

    # if list is empty fill it with default diseases
    if len(disease_list) == 0:
        print(f"ERROR: No diseases in disease_file")
        sys.exit(0)

    print("========================")
    print("Diseases: ", disease_list)
    print("========================")

    return disease_list

# =======================================
#   N E T W O R K   C O M P A R I S O N
# =======================================
def compare_networks(diseases):

    # Create Human-Human Interactome network
    biogrid = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"
    hhi_df  = select_hhi_only(biogrid)
    hhi     = nx.from_pandas_edgelist(hhi_df,
                                      source = "Official Symbol Interactor A",
                                      target = "Official Symbol Interactor B",
                                      create_using=nx.Graph())  #nx.Graph doesn't allow duplicated edges
    # Remove self loops
    self_loop_edges = list(nx.selfloop_edges(hhi))
    hhi.remove_edges_from(self_loop_edges)

    # Largest Connected Component
    LCC_hhi = isolate_LCC(hhi)
    LCC_hhi = hhi.subgraph(LCC_hhi).copy()

    # Define attributes
    attributes = ["number_of_disease_genes",
                  "number_of_connected_components"
                  "LCC_size",
                  "density",
                  "percentage_of_disease_genes_in_LCC",
                  "longest_path_length_in_LCC",
                  "longest_path_lenght_in_interactome",
                  "average_path_length"
                  "degree_distribution",
                  "average_degree",
                  "clustering_coefficient",
                  "modularity"
                  "global_efficency",
                  "assortativity"]

    # Init disease attributes dictionary
    disease_attributes_dictionary = {}

    # Fill the dictionary
    for disease in diseases:
        disease_attributes_dictionary[disease] = []

        # number_of_disease_genes
        disease_genes = get_disease_genes_from_gda("data/curated_gene_disease_assosiactions.tsv")
        disease_attributes_dictionary[disease].append(len(disease_genes))

        # number_of_connected_components
        num_connected_components = get_disease_num_connected_components(hhi_df, disease, from_curated=True)
        disease_attributes_dictionary[disease].append(num_connected_components)

        # LCC_size
        disease_LCC = get_disease_LCC(hhi_df, disease, from_curated=True)
        disease_LCC = LCC_hhi.subgraph(disease_LCC).copy()

        disease_attributes_dictionary[disease].append(nx.number_of_nodes(disease_LCC))

        # density
        disease_density = get_density(disease_LCC)
        disease_attributes_dictionary[disease].append(disease_density)

        # TODO: continue following the attributes order



        # for attribute in attributes:
        #     disease_attributes_dictionary[disease].append("test")

    # Save the dictionary as CSV fils
    df = pd.DataFrame.from_dict(disease_attributes_dictionary, orient='index',
                      columns=attributes)

    df.to_csv("tables/top_performing_diseases_analysis.csv")


# ============
#   M A I N
# ============

if __name__ == "__main__":

    # Read input
    args = parse_args()
    diseases = read_terminal_input(args)

    # Compare disease networks before enrichment
    compare_networks(diseases)

