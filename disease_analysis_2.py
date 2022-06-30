import argparse
import csv
from turtle import title

import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
from graph_tiger.measures import run_measure
from sklearn.preprocessing import robust_scale
from torch import absolute

from utils.data_utils import *
from utils.network_utils import *

# =======================
#   R E A D   I N P U T
# =======================

def print_usage():
    print(' ')
    print('        usage: python3 disease_analysis_2.py --database --disease_file ')
    print('        -----------------------------------------------------------------')
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison')
    print('                                   (default: "data/disease_file.txt).')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Get algorithms to compare and on which validation.')
    parser.add_argument('--disease_file', type=str, default="data/diamond_dataset/diseases.txt",
                    help='Relative path to the file containing the disease names to use for the analysis (default: "data/diamond_dataset/diseases.txt")')
    parser.add_argument('--database', type=str, default="diamond_dataset",
                    help='Name of the database to use for the disease analyisis. (default: "diamond_dataset")')
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
    disease_file = args.disease_file
    database = args.database

    # Get disease list from file
    try:
        diseases = read_disease_file(disease_file)
    except:
        print(f"Not found file in {disease_file} or no valid location.")
        sys.exit(0)

    # if list is empty fill it with default diseases
    if len(diseases) == 0:
        print(f"ERROR: No diseases in disease_file")
        sys.exit(0)

    # Check database
    if database not in ["biogrid", "stringdb", "pnas", "diamond_dataset"]:
        print("ERROR: no valid database name")
        print_usage()
        sys.exit(1)

    if database == "biogrid":
        database_path = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"

    if database == "stringdb":
        database_path = "data/9606.protein.links.full.v11.5.txt"

    if database == "pnas":
        database_path = "data/pnas.2025581118.sd02.csv"
    
    if database == "diamond_dataset":
        database_path = "data/diamond_dataset/Interactome.tsv"

    print("========================")
    print("Diseases: ", diseases)
    print("Database: ", database)
    print("========================")

    return diseases, database, database_path

# ======================================
#    N E T W O R K   A N A L Y S I S
# ======================================
def analyze_disease_networks(disease_networks, database_name = None, enriching_algorithm=None, algorithm_predicted_genes = {}):
    '''
    NB: predicted_genes given in input are enriched genes, if there are any. 
        When call this function after enrichment anlysis, pass also the dictioanry of predicted genes 
    '''


    # Define attributes
    attributes = ["number_of_disease_genes",
                  "number_of_connected_components",
                  "percentage_of_connected_components",
                  "LCC_size",
                  "density",
                  "density_of_LCC",
                  "percentage_of_disease_genes_in_LCC",
                #   "longest_path_length_in_LCC",
                #   "longest_path_lenght_in_interactome",
                  "average_path_length",
                  "diameter_of_LCC",
                  "average_degree",
                  "clustering_coefficient",
                  "modularity",
                  "global_efficency",
                  "assortativity",
                  "community",
                  'node_connectivity',              # robustness parameters: from here
                  'edge_connectivity',              #
                  'spectral_radius',                #
                  'spectral_gap',                   #
                  'natural_connectivity',           #
                  'algebraic_connectivity',         #
                  'num_spanning_trees'            #
                #   'effective_resistance',           #
                   #'generalized_robustness_index',   # to here
                #   'small-world'
                  ]


    # Init disease attributes dictionary
    disease_attributes_dictionary = {}

    # Fill the dictionary
    i = 0
    for disease, disease_network in disease_networks.items():
        try: 
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

            # # 8. longest_path_in_LCC
            # #if we don't have predicted genes for enrichment analysis
            # if (algorithm_predicted_genes == {}) :
            #     disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_LCC(disease))
            # #if we are in enrichment analysis
            # else:
            #     disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_LCC(disease, algorithm_predicted_genes[disease]))

            # # 9. longest_path_in_interactome
            # #if we don't have predicted genes for enrichment analysis
            # if (algorithm_predicted_genes == {}) :
            #     disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_interactome(disease))
            # #if we are in enrichment analysis
            # else:
            #     disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_interactome(disease, algorithm_predicted_genes[disease]))


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
            modularity = nx_comm.modularity(disease_network, nx_comm.label_propagation_communities(disease_network))
            disease_attributes_dictionary[disease].append(modularity)

            # 15. global_efficency
            disease_attributes_dictionary[disease].append(nx.global_efficiency(disease_network))

            # 16. assortativity
            disease_attributes_dictionary[disease].append(nx.degree_assortativity_coefficient(disease_network))

            # 17. Communities greater than 1
            communities = nx.algorithms.community.greedy_modularity_communities(disease_network)
            # compute how many communities are greater than one. Smaller ones are meaningless.
            communitites_greater_than_1 = 0
            for community in communities:
                if len(community) > 1:
                    communitites_greater_than_1 +=1
            disease_attributes_dictionary[disease].append(communitites_greater_than_1)

            # 18. Node connectivity
            # node_connectivity = run_measure(disease_network, measure='node_connectivity')
            node_connectivity = nx.node_connectivity(disease_network)
            disease_attributes_dictionary[disease].append(node_connectivity)

            # 19. Edge connectivity
            # edge_connectivity = run_measure(disease_network, measure='edge_connectivity')
            edge_connectivity = nx.edge_connectivity(disease_network)
            disease_attributes_dictionary[disease].append(edge_connectivity)

            # 20. Spectral radius
            spectral_radius = run_measure(disease_network, measure='spectral_radius')
            disease_attributes_dictionary[disease].append(spectral_radius)

            # 21. Spectral gap
            spectral_gap = run_measure(disease_network, measure='spectral_gap')
            disease_attributes_dictionary[disease].append(spectral_gap)

            # 22. Natural connectivity
            natural_connectivity = run_measure(disease_network, measure='natural_connectivity')
            disease_attributes_dictionary[disease].append(natural_connectivity)

            # 23. Algebraic connectivity
            algebraic_connectivity = run_measure(disease_network, measure='algebraic_connectivity')
            disease_attributes_dictionary[disease].append(algebraic_connectivity)

            # 24. Number of spanning trees
            number_spanning_trees = run_measure(disease_network, measure='number_spanning_trees')
            disease_attributes_dictionary[disease].append(number_spanning_trees)

            # # 25. Effective resistance
            # effective_resistance = run_measure(disease_network, measure='effective_resistance')
            # disease_attributes_dictionary[disease].append(effective_resistance)

            # # 26. Generalized robustness index (IT SHOULD RETURN ALWAYS NONE)
            # generalized_robustness_index = run_measure(disease_network, measure='generalized_robustness_index')
            # disease_attributes_dictionary[disease].append(generalized_robustness_index)

            # 27. Small-world
            # disease_attributes_dictionary[disease].append(nx.sigma(disease_LCC))
        except:
            print("ERROR with disease: ", disease)
            continue

        # Export the network for Cytoscape visualization
        if enriching_algorithm is not None:
            nx.write_graphml(disease_network, f"plots/networks/graphml/{string_to_filename(disease)}.{enriching_algorithm}_enriched.graphml")
        else:
            nx.write_graphml(disease_network, f"plots/networks/graphml/{string_to_filename(disease)}.graphml")


    # Save the dictionary as CSV fils
    df = pd.DataFrame.from_dict(disease_attributes_dictionary, orient='index',
                      columns=attributes)

    if enriching_algorithm is not None:
        df.to_csv(f"tables/diseases_analysis__{enriching_algorithm}_enriched__{database_name}.csv")
    else:
        df.to_csv(f"tables/diseases_analysis__{database_name}.csv")

    return df

def compare_network_attributes(N1, N2, nname1=None, nname2=None, heatmap=False, outfile=None):

    # Check that N1 and N2 have the same labels
    assert(list(N1.index) == list(N2.index))
    assert(list(N1.columns) == list(N2.columns))

    rows = list(N1.index)
    columns = list(N1.columns)

    # Init DataFrame
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
                if N1_attribute_value == 0:
                        N1_attribute_value = 0.001

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
                            vmin=-100, vmax=100)

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
    diseases, database_name, database_path = read_terminal_input(args)

    # Build the Human-Human Interactome
    if database_name == "biogrid":
        hhi = build_network_from_biogrid(database_path,
                                        hhi_only=True,
                                        physical_only=True,
                                        remove_self_loops=True)
    if database_name == "stringdb":
        hhi = build_network_from_stringdb(database_path,
                                          remove_self_loops=True)

    if database_name == "pnas":
        hhi = build_network_from_pnas(database_path,
                                      remove_self_loops=True)

    if database_name == "diamond_dataset":
        hhi = build_network_from_diamond_dataset(database_path,
                                                remove_self_loops=True)

    # Isolate the Largest Connected Component
    hhi_lcc = LCC(hhi)

    # ******************************
    #   Build the Disease Networks
    # ******************************

    # Dictionary of networks
    original_disease_networks = {}


    gda_curated = "data/curated_gene_disease_associations.tsv"
    seeds_file = "data/diamond_dataset/seeds.tsv"

    # Build the network for each disease and append it to the list
    for disease in diseases:

        if database_name in ["diamond_dataset"]:
            disease_genes = get_disease_genes_from_seeds_file(seeds_file, disease, fix_random=True)
        else:
            disease_genes = get_disease_genes_from_gda(gda_curated, disease, translate_in=database_name)
                
        # check that the list of disease genes is not empty
        if len(disease_genes) == 0:
            print(f"WARNING: {disease} has no disease genes. Skip this disease")
            continue

        # Extract the subgraph from the HHI LCC and remove the self-loops
        disease_network = hhi_lcc.subgraph(disease_genes).copy()
        disease_network.remove_edges_from(nx.selfloop_edges(disease_network))

        # Add it to the dictionary of networks
        original_disease_networks[disease] = disease_network

    # print("List of Disease Networks: ", disease_networks)

    orginal_network_attributes = analyze_disease_networks(original_disease_networks, database_name=database_name)
    
    '''
    # ***************************************************
    #   Perform enrichment through DIAMOnD and pDIAMOnD
    # ***************************************************

    DIAMOnD_enriched_networks = {}      # Dict of disease networks enriched with DIAMOnD
    pDIAMOnD_enriched_networks = {}     # Dict of disease networks enriched with pDIAMOnD
    pDIAMOnD_alternative_enriched_networks = {}
    
    #define dictionaries which contain <key: disease, value: list of enriched genes for the disease>
    diamond_predicted_genes = {} 
    pdiamond_predicted_genes = {}
    pdiamond_alternative_predicted_genes = {}

    for disease in diseases:
        disease_genes = get_disease_genes_from_gda("data/curated_gene_disease_associations.tsv", disease, training_mode=True)

        # DIAMOnD
        print(f"\nPerforming the enrichemt of {disease} with DIAMOnD\n")

        added_nodes = DIAMOnD(LCC_hhi, disease_genes, 25, 1, outfile="tmp/diamond_output.txt")
        predicted_genes = [item[0] for item in added_nodes]
        diamond_predicted_genes[disease] = predicted_genes      #add predicted genes to the dictionary of predicted genes
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
        pdiamond_predicted_genes[disease] = predicted_genes  #add predicted genes to the dictionary of predicted genes
        pDIAMOnD_enriched_genes = disease_genes + predicted_genes

        print(f"num_orginal_genes: {len(disease_genes)} genes")
        print(f"num_predicted_genes: {len(predicted_genes)} genes")
        print(f"total_enriched_genes: {len(pDIAMOnD_enriched_genes)} genes")

        # Extract the subgraph from the HHI LCC and remove the self-loops
        disease_network = LCC_hhi.subgraph(pDIAMOnD_enriched_genes).copy()
        disease_network.remove_edges_from(nx.selfloop_edges(disease_network))

        # Add it to the dictionary of pDIAMOnD enriched networks
        pDIAMOnD_enriched_networks[disease] = disease_network

        #pDIAMOnD_alternative
        print(f"\nPerforming the enrichemt of {disease} with pDIAMOnD_alternative\n")

        added_nodes = pDIAMOnD_alternative(LCC_hhi, disease_genes, 25, 1, outfile="tmp/pdiamond_output.txt")
        predicted_genes = [item[0] for item in added_nodes]
        pdiamond_alternative_predicted_genes[disease] = predicted_genes  #add predicted genes to the dictionary of predicted genes
        pDIAMOnD_alternative_enriched_genes = disease_genes + predicted_genes

        print(f"num_orginal_genes: {len(disease_genes)} genes")
        print(f"num_predicted_genes: {len(predicted_genes)} genes")
        print(f"total_enriched_genes: {len(pDIAMOnD_alternative_enriched_genes)} genes")

        # Extract the subgraph from the HHI LCC and remove the self-loops
        disease_network = LCC_hhi.subgraph(pDIAMOnD_alternative_enriched_genes).copy()
        disease_network.remove_edges_from(nx.selfloop_edges(disease_network))

        # Add it to the dictionary of pDIAMOnD enriched networks
        pDIAMOnD_alternative_enriched_networks[disease] = disease_network

    # Compare disease networks AFTER the enrichment
    DIAMOnD_enriched_network_attributes = analyze_disease_networks(DIAMOnD_enriched_networks, enriching_algorithm="DIAMOnD", algorithm_predicted_genes = diamond_predicted_genes)
    pDIAMOnD_enriched_network_attributes = analyze_disease_networks(pDIAMOnD_enriched_networks, enriching_algorithm="pDIAMOnD", algorithm_predicted_genes = pdiamond_predicted_genes)
    pDIAMOnD_alternative_enriched_network_attributes = analyze_disease_networks(pDIAMOnD_enriched_networks, enriching_algorithm="pDIAMOnD_alternative", algorithm_predicted_genes = pdiamond_alternative_predicted_genes)


    # ***********************************
    #   How much the attributes growed
    # ***********************************
    compare_network_attributes(orginal_network_attributes, DIAMOnD_enriched_network_attributes, nname1="Orginial", nname2="DIAMOnD", heatmap=True, outfile="tables/percentage_differences.original_vs_DIAMOnD_enriched_networks.csv")
    compare_network_attributes(orginal_network_attributes, pDIAMOnD_enriched_network_attributes, nname1="Orginial", nname2="pDIAMOnD", heatmap=True, outfile="tables/percentage_differences.original_vs_pDIAMOnD_enriched_networks.csv")
    compare_network_attributes(DIAMOnD_enriched_network_attributes, pDIAMOnD_enriched_network_attributes, nname1="DIAMOnD", nname2="pDIAMOnD", heatmap=True, outfile="tables/percentage_differences.DIAMOnD_vs_pDIAMOnD_enriched_networks.csv")
    compare_network_attributes(DIAMOnD_enriched_network_attributes, pDIAMOnD_alternative_enriched_network_attributes, nname1="DIAMOnD", nname2="pDIAMOnD_alternative", heatmap=True, outfile="tables/percentage_differences.DIAMOnD_vs_pDIAMOnD_alternative_enriched_networks.csv")
    '''
