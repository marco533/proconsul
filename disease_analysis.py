import argparse
import csv
from turtle import title

import networkx as nx
import numpy as np
import pandas as pd
#from torch import absolute
#import seaborn as sns
from matplotlib import pyplot as plt

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

# ======================================
#    N E T W O R K   A N A L Y S I S
# ======================================
def analyze_disease_networks(disease_networks, enriching_algorithm=None):

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
                  "diameter_of_LCC",
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
        disease_LCC = disease_network.subgraph(disease_LCC).copy()
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
        disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_LCC(disease))

        # 9. longest_path_in_interactome
        disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_interactome(disease))

        # 10. average_path_length
        # disease_attributes_dictionary[disease].append(nx.average_shortest_path_length(disease_LCC))
        shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(disease_network))
        average_path_lengths = [np.mean(list(spl.values())) for spl in shortest_path_lengths.values()]
        disease_attributes_dictionary[disease].append(np.mean(average_path_lengths))

        # 11. diameter_of_LCC
        disease_attributes_dictionary[disease].append(max(nx.eccentricity(disease_LCC, sp=shortest_path_lengths).values()))

        # 12. average_degree
        # disease_attributes_dictionary[disease].append(nx.average_degree_connectivity(disease_network))
        disease_attributes_dictionary[disease].append(np.mean([d for _, d in disease_network.degree()]))

        # 13. clustering_coefficient
        disease_attributes_dictionary[disease].append(nx.average_clustering(disease_network))

        # 14. modularity
        disease_attributes_dictionary[disease].append("test")

        # 15. global_efficency
        disease_attributes_dictionary[disease].append(nx.global_efficiency(disease_network))

        # 16. assortativity
        disease_attributes_dictionary[disease].append(nx.degree_assortativity_coefficient(disease_network))

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

    return df

def compare_network_attributes(N1, N2, nname1=None, nname2=None, heatmap=False, outfile=None):

    # Check that N1 and N2 have the same labels
    assert(list(N1.index) == list(N2.index))
    assert(list(N1.columns) == list(N2.columns))

    rows = list(N1.index)
    columns = list(N1.columns)

    # Init two DataFrames
    # 1. Percentage differences
    # 2. Absolute values
    percentage_differences  = pd.DataFrame(np.zeros((len(rows), len(columns))),
                                            index=rows,
                                            columns=columns)

    for r in rows:
        for c in columns:
            # Get attribute value
            N1_attribute_value = N1.at[r, c]
            N2_attribute_value = N2.at[r, c]

            # Percentage difference
            if type(N1_attribute_value) == str or type(N2_attribute_value) == str:
                percentage_diff = 0.0
            else:
                if N1_attribute_value == 0: N1_attribute_value = 0.001
                percentage_diff = (N2_attribute_value - N1_attribute_value) * 100 / N1_attribute_value

            # Save the percentage value
            percentage_differences.at[r, c] = percentage_diff

    # Save the DataFrame as CSV file
    percentage_differences.to_csv(outfile)

    # Visualize the difference of the attributes as an heatmap
    if heatmap:
        # Fix the output filename
        outfile = outfile.replace("tables/","plots/heatmaps/network_attributes/").replace(".csv", ".png")

        # Plot heatmap
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        hm = sns.heatmap(percentage_differences,
                            annot=False,
                            cmap=cmap,
                            cbar_kws={"label": f"<--{nname1}  |  {nname2}-->"},
                            center=0,
                            vmin=-100, vmax=100)#.set(title=f"Percentage Differences | {nname2} Vs {nname1} enriched networks")

        hm.set_title(f"Percentage Differences | {nname2} Vs {nname1} enriched networks")

        figure = hm.get_figure()
        figure.savefig(outfile, bbox_inches='tight')

        # Close previous plots
        plt.close()

    return 0


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
    original_disease_networks = {}

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
        original_disease_networks[disease] = disease_network

    # print("List of Disease Networks: ", disease_networks)

    orginal_network_attributes = analyze_disease_networks(original_disease_networks)

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
    DIAMOnD_enriched_network_attributes = analyze_disease_networks(DIAMOnD_enriched_networks, enriching_algorithm="DIAMOnD")
    pDIAMOnD_enriched_network_attributes = analyze_disease_networks(pDIAMOnD_enriched_networks, enriching_algorithm="pDIAMOnD")

    # ***********************************
    #   How much the attributes growed
    # ***********************************
    compare_network_attributes(orginal_network_attributes, DIAMOnD_enriched_network_attributes, nname1="Orginial", nname2="DIAMOnD", heatmap=True, outfile="tables/percentage_differences.original_vs_DIAMOnD_enriched_networks.csv")
    compare_network_attributes(orginal_network_attributes, pDIAMOnD_enriched_network_attributes, nname1="Orginial", nname2="pDIAMOnD", heatmap=True, outfile="tables/percentage_differences.original_vs_pDIAMOnD_enriched_networks.csv")
    compare_network_attributes(DIAMOnD_enriched_network_attributes, pDIAMOnD_enriched_network_attributes, nname1="DIAMOnD", nname2="pDIAMOnD", heatmap=True, outfile="tables/percentage_differences.DIAMOnD_vs_pDIAMOnD_enriched_networks.csv")



