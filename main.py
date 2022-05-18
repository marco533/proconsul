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
    print('        usage: python3 project.py --algs --validation --disease_file --diffusion_time --num_iters_pdiamond --pdiamond_mode')
    print('        -----------------------------------------------------------------')
    print('        algs                     : List of algorithms to use to collect results.')
    print('                                   They can be: "diamond", "pdiamond", "pdiamond_rank", "pdiamond_temp", "pdiamond_topk", "pdiamond_all", "heat_diffusion"')
    print('                                   If all, run all the algorithms. (default: all')
    print('        validation               : Type of validation on which test the algorithms. It can be')
    print('                                   "kfold", "extended" or "all".')
    print('                                    If all, perform both the validations. (default: all')
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison')
    print('                                   (default: "data/disease_file.txt).')
    print('        diffusion_time           : Diffusion time for heat_diffusion algorithm.')
    print('                                   (default: 0.005)')
    print('        num_iters_pdiamond       : Number of iteration for pDIAMOnD.')
    print('                                   (default: 10)')
    print('        pdiamond_mode            : Run the classic or the alternative version of pdiamond')
    print('                                   (default: classic)')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set disease, algorithms and validation')
    parser.add_argument('-a','--algs', nargs='+', default=["diamond", "pdiamond", "pdiamond_temp", "pdiamond_rank", "pdiamond_topk", "pdiamond_all", "heat_diffusion"],
                    help='List of algorithms to run (default: all)')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation. (default: all')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Relative path to the file with disease names (default: "data/disease_file.txt)')
    parser.add_argument('--diffusion_time', type=float, default=0.005,
                    help='Diffusion time for heat_diffusion algorithm. (default: 0.005')
    parser.add_argument('--num_iters_pdiamond', type=int, default=10,
                    help='Number of iteration for pDIAMOnD. (default: 10)')
    parser.add_argument('--pdiamond_mode', type=str, default="classic",
                    help='pDIAMOnD mode (default: classic)')
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
    diffusion_time      = args.diffusion_time
    num_iters_pdiamond  = args.num_iters_pdiamond
    pdiamond_mode       = args.pdiamond_mode

    # Check algorithm names
    for alg in algs:
        if alg not in ["diamond", "pdiamond", "pdiamond_rank", "pdiamond_temp", "pdiamond_topk", "pdiamond_all", "heat_diffusion"]:
            print(f"ERROR: {alg} is not a valid algorithm!")
            print_usage()
            sys.exit(0)

    # Check validation
    if validation not in ["kfold", "extended", "all"]:
        print(f"ERROR: {validation} is no valid validation method!")
        print_usage()
        sys.exit(0)

    # Get diesease list from file
    try:
        disease_list = read_disease_file(disease_file)
    except:
        print(f"Not found file in {disease_file} or no valid location.")
        sys.exit(0)

    # if empty, exit with error
    if len(disease_list) == 0:
        print(f"ERROR: No diseases in disease_file")
        sys.exit(1)

    # Get list of validations
    if validation == 'all':
        validation_list = ['kfold', 'extended']
    else:
        validation_list = [validation]

    # Check diffusion time
    if diffusion_time < 0:
        print(f"ERROR: diffusion time must be greater than 0")
        sys.exit(0)

    # Check num_iters_pdiamond
    if num_iters_pdiamond <= 0:
        print(f"ERROR: num_iters_pdiamond must be greater or equal of 1")
        sys.exit(0)

    print('')
    print(f"============================")

    print(f"Algorithm: {algs}")
    print(f"Validations: {validation_list}")
    print(f"Diseases: {len(disease_list)}")
    print(f"Diffusion Time: {diffusion_time}")
    print(f"Num iterations pDIAMOnD: {num_iters_pdiamond}")

    print(f"============================")
    print('')


    return algs, validation_list, disease_list, diffusion_time, num_iters_pdiamond


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
