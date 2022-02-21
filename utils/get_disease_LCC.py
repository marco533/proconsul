'''
With this script we, given a disease list, we get the size of the Largest Connected
Component of the disease Module inside the Human Human Interactome.
'''

import argparse
import csv
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

from data_utils import *

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Get disease file and output file where to save the LCCs info')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Position to disease file (default: "data/disease_file.txt)')
    parser.add_argument('--output_file', type=str, default="results/disease_LCC_info.csv",
                    help='Where to save the LCC info (default: "results/disease_LCC_info.csv")')
    return parser.parse_args()

def read_terminal_input(args):
    '''
    Read the arguments passed by command line.
    '''

    # read the parsed values
    disease_file    = args.disease_file
    output_file     = args.output_file

    print('')
    print(f"============================")
    print(f"disease_file: {disease_file}")
    print(f"output_file: {output_file}")
    print(f"============================")
    print('')

    return disease_file, output_file

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


# ================== #
#  CALL BY FUNCTION  #
# ================== #

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

# ===================== #
#  CALL BY COMMAND LINE #
# ===================== #

if __name__ == "__main__":
    # parse terminal input
    args = parse_args()
    disease_file, output_file = read_terminal_input(args)

    # init the output file
    with open(output_file, 'w') as of:
        writer = csv.writer(of)
        writer.writerow(["DISEASE NAME", "NUM CONNECTED COMPONENTS", "SIZE LCC"])


    # select the human-human interactions from biogrid
    biogrid_file = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"
    hhi_df = select_hhi_only(biogrid_file)

    # from this datafram select only the disease genes
    disease_list = get_diseases_from_file(disease_file)
    for disease in disease_list:
        disease_df = select_disease_interactions_only(hhi_df, disease)

        # create the network from the disease dataframe
        disease_network = nx.from_pandas_edgelist(disease_df,
                                                  source = "Official Symbol Interactor A",
                                                  target = "Official Symbol Interactor B",
                                                  create_using=nx.Graph())  #x.Graph doesn't allow duplicated edges

        # remove self loops
        self_loop_edges = list(nx.selfloop_edges(disease_network))
        disease_network.remove_edges_from(self_loop_edges)

        # save disease network information on the output file
        with open(output_file, 'a') as of:
            writer = csv.writer(of)
            # find connected components
            conn_comp = list(nx.connected_components(disease_network))

            # sort the connected components by descending size
            conn_comp_len = [len(c) for c in sorted(conn_comp, key=len, reverse=True)]

            # isolate the LCC
            LCC = max(conn_comp, key=len)

            writer.writerow([disease, len(conn_comp), len(LCC)])

    print("DONE! Output in the results folder")
