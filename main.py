# imports
import argparse
import sys
from test.extended_validation import *
from test.kfold_validation import *

import networkx as nx
import numpy as np
import pandas as pd

from utils.network_utils import *
from utils.data_utils import *


def print_usage():
    print(' ')
    print('        usage: python3 project.py --algorithm --validation --disease_file')
    print('        -----------------------------------------------------------------')
    print('        disease_file     : Position of a txt file containig a disease name for each line')
    print('                           (default: "data/disease_file.txt"')
    print('        algorithm        : Algorithm to test. It can be "diamond", "prob_diamond or "all".')
    print('                           If all, run both the algorithms. (default: all')
    print('        validation       : type of validation on which test the algorithms. It can be')
    print('                           "kfold", "extended" or "all".')
    print('                           If all, perform both the validations. (default: all')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set disease, algorithm and validation')
    parser.add_argument('--algorithm', type=str, default='all',
                    help='Algorithm to test. (default: all)')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation. (default: all')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Position to disease gile (default: "data/disease_file.txt)')
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

    # read the parsed values
    algorithm       = args.algorithm
    validation      = args.validation
    disease_file    = args.disease_file

    print('')
    print(f"============================")

    print(f"algorithm: {algorithm}")
    print(f"validation: {validation}")
    print(f"disease_file: {disease_file}")

    # get disease list from file
    try:
        disease_list = read_disease_file(disease_file)
    except:
        print(f"Not found file in {disease_file} or no valid location.")
        sys.exit(0)
        return

    # if list is empty fill it with default diseases
    if len(disease_list) == 0:
        print(f"ERROR: No diseases in disease_file")
        sys.exit(0)

    print(f"============================")
    print('')

    # check if is a valid algorithm
    if algorithm not in ["diamond", "prob_diamond", "diable", "moses", "markov_clustering", "heat_diffusion", "RWR", "all"]:
        print(f"ERROR: {algorithm} is no valid algorithm!")
        print_usage()
        sys.exit(0)

    # check if is a valid validation
    if validation not in ["kfold", "extended", "all"]:
        print(f"ERROR: {validation} is no valid validation method!")
        print_usage()
        sys.exit(0)

    return disease_list, algorithm, validation


# main
if __name__ == "__main__":

    # ============ #
    #  READ INPUT  #
    # ============ #

    args = parse_args()
    disease_list, algorithm, validation = read_terminal_input(args)

    # ================ #
    #  CREATE NETWORK  #
    # ================ #

    # select the human-human interactions from biogrid
    biogrid_file = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"
    hhi_df = select_hhi_only(biogrid_file)

    # create the hhi from the filtered data frame
    hhi = nx.from_pandas_edgelist(hhi_df,
                                  source = "Official Symbol Interactor A",
                                  target = "Official Symbol Interactor B",
                                  create_using=nx.Graph())  #x.Graph doesn't allow duplicated edges
    print("Network Info:")
    print(nx.info(hhi), end="\n\n")

    # remove self loops
    self_loop_edges = list(nx.selfloop_edges(hhi))
    hhi.remove_edges_from(self_loop_edges)
    print("After removing self-loops:")
    print(nx.info(hhi), end="\n\n")

    # isolate the largest connected component
    LCC_hhi = isolate_LCC(hhi)

    print("Isolating the LCC:")
    print(nx.info(LCC_hhi), end="\n\n")

    # =================== #
    #  K-FOLD VALIDATION  #
    # =================== #

    gda_filename = "data/curated_gene_disease_associations.tsv"

    if validation == "kfold" or validation == "all":
        for disease in disease_list:

            # get disease genes from curated GDA
            disease_genes = get_disease_genes_from_gda(gda_filename, disease)

            # run the k-fold validation on {algorithm}
            k_fold_cross_validation(LCC_hhi, disease_genes, algorithm, disease, K=5, num_iters_prob_diamond=1)

            # # MCL: Best inflation valure
            # best_inflation = get_best_inflation_value(LCC_hhi)

    # ===================== #
    #  EXTENDED VALIDATION  #
    # ===================== #

    all_gda_filename = "data/all_gene_disease_associations.tsv"

    if validation == "extended" or validation == "all":
        for disease in disease_list:

            # get disease genes from curated and all GDA
            curated_disease_genes = get_disease_genes_from_gda(gda_filename, disease)
            all_disease_genes = get_disease_genes_from_gda(all_gda_filename, disease)

            # remove from all the genes that are already in curated
            new_disease_genes = list(set(all_disease_genes) - set(curated_disease_genes))

            # run the extended validation on {algorithm}
            extended_validation(LCC_hhi, curated_disease_genes, new_disease_genes, algorithm, disease, num_iters_prob_diamond=1)
