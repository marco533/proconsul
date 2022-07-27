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
    print('        usage: python3 analyze_disease_network.py --database --disease_file ')
    print('        --------------------------------------------------------------------')
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison')
    print('                                   (default: "data/disease_file.txt).')
    print('        database                 : Name of the database to use for the disease analyisis.')
    print('                                   (default: "diamond_dataset")')
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
def analyze_disease_networks(disease_networks, database_name=None, interactome=None, enriching_algorithm=None, algorithm_predicted_genes = {},
                            skip_path_lengths=False, skip_clustering=False, skip_global_efficency=False, skip_assortativity=False,
                            skip_communities=False, skip_others=False, outfile=None):
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
                  "longest_shortest_path_in_interactome",
                  "longest_shortest_path_in_LCC",
                  "average_path_length",
                  "diameter_of_LCC",
                  "average_degree",
                  "clustering_coefficient",
                  "modularity",
                  "global_efficency",
                  "assortativity",
                  "number_of_community",
                  'node_connectivity',              # robustness parameters: from here
                  'edge_connectivity',              #
                  'spectral_radius',                #
                  'spectral_gap',                   #
                  'natural_connectivity',           #
                  'algebraic_connectivity',         #
                  'num_spanning_trees'              #
                # 'effective_resistance',           #
                # 'generalized_robustness_index',   # to here
                # 'small-world'
                  ]


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
        print("1. Done")
        # 2. number_of_connected_components
        connected_components = list(nx.connected_components(disease_network))
        number_of_connected_components = len(connected_components)
        disease_attributes_dictionary[disease].append(number_of_connected_components)
        print("2. Done")

        # 3. percentage_of_connected_components
        percentage_connected_components = number_of_connected_components / number_of_disease_genes
        disease_attributes_dictionary[disease].append(percentage_connected_components)
        print("3. Done")

        # 4. LCC_size
        disease_LCC = max(connected_components, key=len)
        disease_LCC = disease_network.subgraph(disease_LCC).copy()
        disease_LCC_size = nx.number_of_nodes(disease_LCC)

        disease_attributes_dictionary[disease].append(disease_LCC_size)
        print("4. Done")


        # 5. density
        density = nx.density(disease_network)
        disease_attributes_dictionary[disease].append(density)
        print("5. Done")

        # 6. density_of_LCC
        density_of_LCC = nx.density(disease_LCC)
        disease_attributes_dictionary[disease].append(density_of_LCC)
        print("6. Done")
        
        # 7. percentage_of_disease_genes_in_LCC
        disease_attributes_dictionary[disease].append(disease_LCC_size / number_of_disease_genes)
        print("7. Done")

        if skip_path_lengths:
            # 8.
            disease_attributes_dictionary[disease].append("Skip")
            print("8. Done")

            # 9.
            disease_attributes_dictionary[disease].append("Skip")
            print("9. Done")

            # 10.
            disease_attributes_dictionary[disease].append("Skip")
            print("10. Done")  

            # 11.
            disease_attributes_dictionary[disease].append("Skip")
            print("11. Done")
        
        else:
            if database_name == "biogrid": # BIOGRID ONLY
                
                # 8. longest_path_in_LCC
                
                # if we don't have predicted genes for enrichment analysis
                if (algorithm_predicted_genes == {}) :
                    disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_LCC(disease))
                # if we are in enrichment analysis
                else:
                    disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_LCC(disease, algorithm_predicted_genes[disease]))
                
                print("8. Done")

                # 9. longest_path_in_interactome
                # if we don't have predicted genes for enrichment analysis
                if (algorithm_predicted_genes == {}) :
                    disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_interactome(disease))
                # if we are in enrichment analysis
                else:
                    disease_attributes_dictionary[disease].append(get_longest_path_for_a_disease_interactome(disease, algorithm_predicted_genes[disease]))

                print("9. Done")

            else:

                # 8. longest simple shortest path in the disease network
                longest_sp_in_the_interactome = 0

                sources = disease_network.nodes()
                targets = disease_network.nodes()
                for s in sources:
                    for t in targets:
                        dist = nx.shortest_path_length(interactome, source=s, target=t)

                        if dist > longest_sp_in_the_interactome:
                            longest_sp_in_the_interactome = dist

                disease_attributes_dictionary[disease].append(longest_sp_in_the_interactome)

                # 9. longest simple shortest path in the LCC
                shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(disease_network))
                longest_sp_length_in_disease_network = [np.max(list(spl.values())) for spl in shortest_path_lengths.values()]
                disease_attributes_dictionary[disease].append(np.max(longest_sp_length_in_disease_network))
                    
            # 10. average_path_length
            # disease_attributes_dictionary[disease].append(nx.average_shortest_path_length(disease_LCC))
            try:
                average_path_lengths = [np.mean(list(spl.values())) for spl in shortest_path_lengths.values()]
            except:
                shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(disease_network))
                average_path_lengths = [np.mean(list(spl.values())) for spl in shortest_path_lengths.values()]

            disease_attributes_dictionary[disease].append(np.mean(average_path_lengths))
            print("10. Done")

            # 11. diameter_of_LCC            
            disease_attributes_dictionary[disease].append(max(nx.eccentricity(disease_LCC, sp=shortest_path_lengths).values()))
            print("11. Done")

        # 12. average_degree
        # disease_attributes_dictionary[disease].append(nx.average_degree_connectivity(disease_network))
        disease_attributes_dictionary[disease].append(np.mean([d for _, d in disease_network.degree()]))
        print("12. Done")

        # 13. clustering_coefficient
        if skip_clustering:
            disease_attributes_dictionary[disease].append("Skip")
        else:
            disease_attributes_dictionary[disease].append(nx.average_clustering(disease_network))
        
        print("13. Done")

        # 14. modularity
        try:
            modularity = nx_comm.modularity(disease_network, nx_comm.label_propagation_communities(disease_network))
        except:
            modularity = 0

        disease_attributes_dictionary[disease].append(modularity)
        print("14. Done")

        # 15. global_efficency
        if skip_global_efficency:
            disease_attributes_dictionary[disease].append("Skip")
        else:
            disease_attributes_dictionary[disease].append(nx.global_efficiency(disease_network))
        print("15. Done")

        # 16. assortativity
        if skip_assortativity:
            disease_attributes_dictionary[disease].append("Skip")
        else:
            try:
                disease_attributes_dictionary[disease].append(nx.degree_assortativity_coefficient(disease_network))
            except:
                disease_attributes_dictionary[disease].append("None")

        print("16. Done")

        # 17. Communities greater than 1
        if skip_communities:
            disease_attributes_dictionary[disease].append("Skip")
        else:
            try:
                communities = nx.algorithms.community.greedy_modularity_communities(disease_network)
            except:
                communities = []
            
            # compute how many communities are greater than one. Smaller ones are meaningless.
            communitites_greater_than_1 = 0
            for community in communities:
                if len(community) > 1:
                    communitites_greater_than_1 +=1
            disease_attributes_dictionary[disease].append(communitites_greater_than_1)
        
        print("17. Done")

        if skip_others:
            # 18.
            disease_attributes_dictionary[disease].append("Skip")

            # 19.
            disease_attributes_dictionary[disease].append("Skip")

            # 20.
            disease_attributes_dictionary[disease].append("Skip")

            # 21.
            disease_attributes_dictionary[disease].append("Skip")

            # 22.
            disease_attributes_dictionary[disease].append("Skip")

            # 23.
            disease_attributes_dictionary[disease].append("Skip")

            # 24.
            disease_attributes_dictionary[disease].append("Skip")
        
        else:
            # 18. Node connectivity
            node_connectivity = run_measure(disease_network, measure='node_connectivity')
            # node_connectivity = nx.node_connectivity(disease_LCC)
            disease_attributes_dictionary[disease].append(node_connectivity)
            print("18. Done")

            # 19. Edge connectivity
            edge_connectivity = run_measure(disease_network, measure='edge_connectivity')
            # edge_connectivity = nx.edge_connectivity(disease_LCC)
            disease_attributes_dictionary[disease].append(edge_connectivity)
            print("19. Done")

            # 20. Spectral radius
            spectral_radius = run_measure(disease_network, measure='spectral_radius')
            disease_attributes_dictionary[disease].append(spectral_radius)
            print("20. Done")

            # 21. Spectral gap
            spectral_gap = run_measure(disease_network, measure='spectral_gap')
            disease_attributes_dictionary[disease].append(spectral_gap)
            print("21. Done")

            # 22. Natural connectivity
            natural_connectivity = run_measure(disease_network, measure='natural_connectivity')
            disease_attributes_dictionary[disease].append(natural_connectivity)
            print("22. Done")

            # 23. Algebraic connectivity
            algebraic_connectivity = run_measure(disease_network, measure='algebraic_connectivity')
            disease_attributes_dictionary[disease].append(algebraic_connectivity)
            print("23. Done")
        
            # 24. Number of spanning trees
            number_spanning_trees = run_measure(disease_network, measure='number_spanning_trees')
            disease_attributes_dictionary[disease].append(number_spanning_trees)
            print("24. Done")

            """
            # 25. Effective resistance
            effective_resistance = run_measure(disease_network, measure='effective_resistance')
            disease_attributes_dictionary[disease].append(effective_resistance)

            # 26. Generalized robustness index (IT SHOULD RETURN ALWAYS NONE)
            generalized_robustness_index = run_measure(disease_network, measure='generalized_robustness_index')
            disease_attributes_dictionary[disease].append(generalized_robustness_index)

            # 27. Small-world
            disease_attributes_dictionary[disease].append(nx.sigma(disease_LCC))
            """
        
        # Export the network for Cytoscape visualization
        if enriching_algorithm is not None:
            nx.write_graphml(disease_network, f"plots/networks/graphml/{string_to_filename(disease)}.{enriching_algorithm}_enriched.graphml")
        else:
            nx.write_graphml(disease_network, f"plots/networks/graphml/{string_to_filename(disease)}.graphml")


    # Save the dictionary as CSV fils
    df = pd.DataFrame.from_dict(disease_attributes_dictionary, orient='index',
                      columns=attributes)

    if outfile is not None:
        df.to_csv(outfile)
        
    else:
        if enriching_algorithm is not None:
            df.to_csv(f"disease_analysis/network_attributes.{enriching_algorithm}_enriched.{database_name}.csv")
        else:
            df.to_csv(f"disease_analysis/network_attributes.{database_name}.csv")

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

def get_neighbors(G, seeds):
    """
    Given a graph G and a list of seed nodes,
    return the neighborus of that seeds.
    """
    neighbors = []

    for node in seeds:
        if not G.has_node(node):
            continue
        
        nn = G.neighbors(node)
        

        for n in nn:
            if n not in neighbors and n not in seeds:
                neighbors.append(n)

    return neighbors


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
    disease_networks_with_neighbors = {}
    only_first_neighbors_networks = {}


    gda_curated = "data/curated_gene_disease_associations.tsv"
    seeds_file = "data/diamond_dataset/seeds.tsv"
    
    # ****************************************
    # Get the first neighbours of the disease 
    # networks and perform again the analysis
    # ****************************************
    
    print("\nBuilding the disease networks...")

    for idx, disease in enumerate(diseases):
        print(f"disease {idx+1}/{len(diseases)}")

        if database_name in ["diamond_dataset"]:
            disease_genes = get_disease_genes_from_seeds_file(seeds_file, disease, fix_random=True)
        else:
            disease_genes = get_disease_genes_from_gda(gda_curated, disease, translate_in=database_name)
                
        # check that the list of disease genes is not empty
        if len(disease_genes) == 0:
            print(f"WARNING: {disease} has no disease genes. Skip this disease")
            continue

        # get the first neighbors
        first_neighbors = get_neighbors(hhi_lcc, disease_genes)

        # concatenate the lists
        disease_genes_with_neighbors = disease_genes + first_neighbors

        # Extract the subgraph from the HHI LCC and remove the self-loops

        # original disease network
        disease_network = hhi_lcc.subgraph(disease_genes)
        # disease_network.remove_edges_from(nx.selfloop_edges(disease_network))

        # disease network + neighbors
        enriched_network = hhi_lcc.subgraph(disease_genes_with_neighbors)
        # enriched_network.remove_edges_from(nx.selfloop_edges(enriched_network))

        # neighbors only
        only_neighbors_net = hhi_lcc.subgraph(first_neighbors)
        # only_neighbors_net.remove_edges_from(nx.selfloop_edges(only_neighbors_net))

        # Add them to the dictionary of networks
        original_disease_networks[disease] = disease_network
        disease_networks_with_neighbors[disease] = enriched_network
        only_first_neighbors_networks[disease] = only_neighbors_net

        

    # get attributes
    print("\nAnalyzing original networks...")
    network_attributes = analyze_disease_networks(original_disease_networks,
                                                    database_name=database_name,
                                                    interactome=hhi_lcc,
                                                    outfile=f"disease_analysis/network_attributes_{database_name}.csv")

    print("\nAnalizing enriched networks with first neighbors...")
    network_attributes_with_first_neighbors = analyze_disease_networks(disease_networks_with_neighbors,
                                                                        database_name=database_name, 
                                                                        interactome=hhi_lcc,
                                                                        outfile=f"disease_analysis/network_attributes_with_first_neighbors_{database_name}.csv")
    
    print("\nAnalizing only first neighbors network")
    only_first_neighbors_attributes = analyze_disease_networks(only_first_neighbors_networks,
                                                                database_name=database_name, 
                                                                interactome=hhi_lcc,
                                                                outfile=f"disease_analysis/only_first_neighbors_attributes_{database_name}.csv")

