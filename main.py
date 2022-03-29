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
    print('        usage: python3 project.py --algorithm --validation --disease_file --diffusion_time --num_iters_pdiamond')
    print('        -----------------------------------------------------------------')
    print('        algorithm                : Algorithm of whic collect the results. It can be "diamond", "pdiamond", "heat_diffusion" or "all".')
    print('                                   If all, run all the algorithms. (default: all')
    print('        validation               : Type of validation on which test the algorithms. It can be')
    print('                                   "kfold", "extended" or "all".')
    print('                                    If all, perform both the validations. (default: all')
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison')
    print('                                   (default: "data/disease_file.txt).')
    print('        diffusion_time           : Diffusion time for heat_diffusion algorithm.')
    print('                                   (default: 0.005)')
    print('        num_iters_pdiamond   : Number of iteration for pDIAMOnD.')
    print('                                   (default: 10)')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set disease, algorithms and validation')
    parser.add_argument('--algorithm', type=str, default='all',
                    help='Algorithm to test. (default: all)')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation. (default: all')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Relative path to the file with disease names (default: "data/disease_file.txt)')
    parser.add_argument('--diffusion_time', type=float, default=0.005,
                    help='Diffusion time for heat_diffusion algorithm. (default: 0.005')
    parser.add_argument('--num_iters_pdiamond', type=int, default=10,
                    help='Number of iteration for pDIAMOnD. (default: 10)')
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
    diffusion_time  = args.diffusion_time
    num_iters_pdiamond = args.num_iters_pdiamond

    # check if is a valid algorithm
    if algorithm not in ["diamond", "pdiamond", "heat_diffusion", "all"]:
        print(f"ERROR: {algorithm} is no valid algorithm!")
        print_usage()
        sys.exit(0)

    # check if is a valid validation
    if validation not in ["kfold", "extended", "all"]:
        print(f"ERROR: {validation} is no valid validation method!")
        print_usage()
        sys.exit(0)

    # get disease list from file
    try:
        disease_list = read_disease_file(disease_file)
    except:
        print(f"Not found file in {disease_file} or no valid location.")
        sys.exit(0)

    # if list is empty fill it with default diseases
    if len(disease_list) == 0:
        print(f"ERROR: No diseases in disease_file")
        sys.exit(0)

    # get the list of validations
    if validation == 'all':
        validation_list = ['kfold', 'extended']
    else:
        validation_list = [validation]

    # get the list of algorithms
    if algorithm == 'all':
        algorithm_list = ["diamond",
                          "pdiamond",
                          "heat_diffusion"]
    else:
        algorithm_list = [algorithm]

    # test diffusion time
    if diffusion_time < 0:
        print(f"ERROR: diffusion time must be greater than 0")
        sys.exit(0)

    # test num_iters_pdiamond
    if num_iters_pdiamond <= 0:
        print(f"ERROR: num_iters_pdiamond must be greater or equal of 1")
        sys.exit(0)

    print('')
    print(f"============================")

    print(f"Algorithm: {algorithm}")
    print(f"Validations: {validation_list}")
    print(f"Diseases: {disease_list}")
    print(f"Diffusion Time: {diffusion_time}")
    print(f"Num iterations pDIAMOnD: {num_iters_pdiamond}")

    print(f"============================")
    print('')


    return algorithm_list, validation_list, disease_list, diffusion_time, num_iters_pdiamond


# main
if __name__ == "__main__":

    # ============ #
    #  READ INPUT  #
    # ============ #

    args = parse_args()
    algorithms, validations, diseases, diffusion_time, num_iters_pdiamond = read_terminal_input(args)

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

    if 'kfold' in validations:
        for alg in algorithms:
            for disease in diseases:

                # get disease genes from curated GDA
                disease_genes = get_disease_genes_from_gda(gda_filename, disease)

                # run the k-fold validation on {algorithm}
                k_fold_cross_validation(LCC_hhi,
                                        disease_genes,
                                        alg,
                                        disease,
                                        K=5,
                                        diffusion_time=diffusion_time,
                                        num_iters_pdiamond=num_iters_pdiamond)

    # ===================== #
    #  EXTENDED VALIDATION  #
    # ===================== #

    all_gda_filename = "data/all_gene_disease_associations.tsv"

    if 'extended' in validations:
        for alg in algorithms:
            for disease in diseases:

                # get disease genes from curated and all GDA
                curated_disease_genes = get_disease_genes_from_gda(gda_filename, disease)
                all_disease_genes = get_disease_genes_from_gda(all_gda_filename, disease)

                # remove from all the genes that are already in curated
                new_disease_genes = list(set(all_disease_genes) - set(curated_disease_genes))

                # run the extended validation on {algorithm}
                extended_validation(LCC_hhi,
                                    curated_disease_genes,
                                    new_disease_genes,
                                    alg,
                                    disease,
                                    diffusion_time=diffusion_time,
                                    num_iters_pdiamond=num_iters_pdiamond)
