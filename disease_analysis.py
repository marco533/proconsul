import argparse
import csv

import networkx as nx
import numpy as np
import pandas as pd

from algorithms.diamond import DIAMOnD
from algorithms.pdiamond import pDIAMOnD, pDIAMOnD_alternative
from utils.data_utils import string_to_filename
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
                    help='Relative path to the file containing the disease names to use for the analysis (default: "data/disease_file.txt)')

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
def compare_disease_networks(disease_networks, enriching_algorithm=None):

    # Define attributes
    attributes = ["number_of_disease_genes",
                  "number_of_connected_components",
                  "percentage_of_connected_components",
                  "LCC_size",
                  "density",
                  "density_of_LCC",
                  "percentage_of_disease_genes_in_LCC",
                  "longest_path_length_in_LCC",
                  "longest_path_lenght_in_interactome",
                  "average_path_length",
                  "degree_distribution",
                  "average_degree",
                  "clustering_coefficient",
                  "modularity",
                  "global_efficency",
                  "assortativity"]

    # Init disease attributes dictionary
    disease_attributes_dictionary = {}

    # Fill the dictionary
    i = 0
    for disease, disease_network in disease_networks.items():
        i += 1
        print(f"disease_network {i} of {len(disease_networks)}: {disease}")

        disease_attributes_dictionary[disease] = []

        # 1. number_of_disease_genes
        number_of_disease_genes = disease_network.number_of_nodes()
        disease_attributes_dictionary[disease].append(number_of_disease_genes)

        # 2. number_of_connected_components
        connected_components = list(nx.connected_components(disease_network))
        number_of_connected_components = len(connected_components)
        disease_attributes_dictionary[disease].append(number_of_connected_components)

        # 3. percentage_of_connected_components
        percentage_connected_components = number_of_connected_components / number_of_disease_genes
        disease_attributes_dictionary[disease].append(percentage_connected_components)

        # 4. LCC_size
        disease_LCC = max(connected_components, key=len)
        disease_LCC = LCC_hhi.subgraph(disease_LCC).copy()
        disease_LCC_size = nx.number_of_nodes(disease_LCC)

        disease_attributes_dictionary[disease].append(disease_LCC_size)

        # 5. density
        density = nx.density(disease_network)
        disease_attributes_dictionary[disease].append(density)

        # 6. density_of_LCC
        density_of_LCC = nx.density(disease_LCC)
        disease_attributes_dictionary[disease].append(density_of_LCC)

        # 7. percentage_of_disease_genes_in_LCC
        disease_attributes_dictionary[disease].append(disease_LCC_size / number_of_disease_genes)

        # 8. longest_path_in_LCC
        # disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_LCC(disease))
        disease_attributes_dictionary[disease].append("test")

        # 9. longest_path_in_interactome
        # disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_interactome(disease))
        disease_attributes_dictionary[disease].append("test")

        # TODO: continue following the attributes order

        # 10. average_path_length
        disease_attributes_dictionary[disease].append("test")

        # 11. degree_distribution
        disease_attributes_dictionary[disease].append("test")

        # 12. average_degree
        disease_attributes_dictionary[disease].append("test")

        # 13. clustering_coefficient
        disease_attributes_dictionary[disease].append("test")

        # 14. modularity
        disease_attributes_dictionary[disease].append("test")

        # 15. global_efficency
        disease_attributes_dictionary[disease].append("test")

        # 16. assortativity
        disease_attributes_dictionary[disease].append("test")

        # Export the network for Cytoscape visualization
        if enriching_algorithm is not None:
            nx.write_graphml(disease_network, f"plots/networks/graphml/{string_to_filename(disease)}.{enriching_algorithm}_enriched.graphml")
        else:
            nx.write_graphml(disease_network, f"plots/networks/graphml/{string_to_filename(disease)}.graphml")


    # Save the dictionary as CSV fils
    df = pd.DataFrame.from_dict(disease_attributes_dictionary, orient='index',
                      columns=attributes)

    if enriching_algorithm is not None:
        df.to_csv(f"tables/top_performing_diseases_analysis.{enriching_algorithm}_enriched.csv")
    else:
        df.to_csv("tables/top_performing_diseases_analysis.csv")


# ============
#   M A I N
# ============

if __name__ == "__main__":

    # Read input
    args = parse_args()
    diseases = read_terminal_input(args)

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

    # ******************************
    #   Build the Disease Networks
    # ******************************

    # Dictionary of networks
    disease_networks = {}

    # Build the network for each disease
    # and append it to the list
    for disease in diseases:
        # Get the disease genes, i.e. the nodes from
        # which we can extract the subgraph from the original interactome
        disease_genes = get_disease_genes_from_gda("data/curated_gene_disease_associations.tsv", disease, training_mode=True)

        # Extract the subgraph from the HHI LCC and remove the self-loops
        disease_network = LCC_hhi.subgraph(disease_genes).copy()
        disease_network.remove_edges_from(nx.selfloop_edges(disease_network))

        # Add it to the dictionary of networks
        disease_networks[disease] = disease_network

    # print("List of Disease Networks: ", disease_networks)

    compare_disease_networks(disease_networks)

    # ***************************************************
    #   Perform enrichment through DIAMOnD and pDIAMOnD
    # ***************************************************

    DIAMOnD_enriched_networks   = {}    # Dict of disease networks enriched with DIAMOnD
    pDIAMOnD_enriched_networks  = {}    # Dict of disease networks enriched with pDIAMOnD

    for disease in diseases:
        disease_genes = get_disease_genes_from_gda("data/curated_gene_disease_associations.tsv", disease, training_mode=True)

        # DIAMOnD
        print(f"\nPerforming the enrichemt of {disease} with DIAMOnD\n")

        added_nodes = DIAMOnD(LCC_hhi, disease_genes, 25, 1, outfile="tmp/diamond_output.txt")
        predicted_genes = [item[0] for item in added_nodes]
        DIAMOnD_enriched_genes = disease_genes + predicted_genes

        print(f"num_orginal_genes: {len(disease_genes)} genes")
        print(f"num_predicted_genes: {len(predicted_genes)} genes")
        print(f"total_enriched_genes: {len(DIAMOnD_enriched_genes)} genes")

        # Extract the subgraph from the HHI LCC and remove the self-loops
        disease_network = LCC_hhi.subgraph(DIAMOnD_enriched_genes).copy()
        disease_network.remove_edges_from(nx.selfloop_edges(disease_network))

        # Add it to the dictionary of DIAMOnD enriched networks
        DIAMOnD_enriched_networks[disease] = disease_network

        # pDIAMOnd
        print(f"\nPerforming the enrichemt of {disease} with pDIAMOnD\n")

        added_nodes = pDIAMOnD(LCC_hhi, disease_genes, 25, 1, outfile="tmp/pdiamond_output.txt")
        predicted_genes = [item[0] for item in added_nodes]
        pDIAMOnD_enriched_genes = disease_genes + predicted_genes

        print(f"num_orginal_genes: {len(disease_genes)} genes")
        print(f"num_predicted_genes: {len(predicted_genes)} genes")
        print(f"total_enriched_genes: {len(pDIAMOnD_enriched_genes)} genes")

        # Extract the subgraph from the HHI LCC and remove the self-loops
        disease_network = LCC_hhi.subgraph(pDIAMOnD_enriched_genes).copy()
        disease_network.remove_edges_from(nx.selfloop_edges(disease_network))

        # Add it to the dictionary of pDIAMOnD enriched networks
        pDIAMOnD_enriched_networks[disease] = disease_network

    # Compare disease networks AFTER the enrichment
    compare_disease_networks(DIAMOnD_enriched_networks, enriching_algorithm="DIAMOnD")
    compare_disease_networks(pDIAMOnD_enriched_networks, enriching_algorithm="pDIAMOnD")
