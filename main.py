# =====================
#     I M P O R T S
# =====================

import argparse
import sys
from test.extended_validation import *
from test.kfold_validation import *

import networkx as nx
import numpy as np
import pandas as pd

from utils.network_utils import *
from utils.data_utils import *




# ===========================
#     R E A D   I N P U T
# ===========================

def print_usage():

    print('                                                                                                                                                         '   )
    print('        usage: python3 project.py --algs --validation --disease_file --database --proconsul_n_rounds --proconsul_temp --proconsul_top_p --proconsul_top_k'   )
    print('        -------------------------------------------------------------------------------------------------------------------------------------------------'   )
    print('        algs                     : List of algorithms to run to collect results.'                                                                            )
    print('                                   They can be: "diamond" or "proconsul" (default: all)'                                                                     )
    print('        validation               : Type of validation on which test the algorithms. It can be'                                                               )
    print('                                   "kfold", "extended" or "all".'                                                                                            )
    print('                                   If all, perform both the validations. (default: all)'                                                                     )
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison.'                                        )
    print('                                   (default: "data/diamond_dataset/diseases.txt).'                                                                                       )
    print('        database                 : Database name from which take the PPIs. Choose from "biogrid", "stringdb", "pnas", or "diamond_dataset".'                 )
    print('                                   (default: "diamond_dataset)'                                                                                              )
    print('        proconsul_n_rounds       : How many different rounds PROCONSUL will do to reduce statistical fluctuation.'                                           )
    print('                                   If you insert a list of values multiple version of PROCONSUL will be run. One for each value.'                            )
    print('                                   (default: 10)'                                                                                                            )
    print('        proconsul_temp           : Temperature value for the PROCONSUL softmax function.'                                                                     )
    print('                                   If you insert a list of values, multiple version of PROCONSUL will be run. One for each value.'                           )
    print('                                   (default: 1.0)'                                                                                                           )
    print('        proconsul_top_p          : Probability threshold value for PROCONSUL nucleus sampling. If 0 no nucleus sampling'                                      )
    print('                                   If you insert a list of values, multiple version of PROCONSUL will be run. One for each value.'                           )
    print('                                   (default: 0.0)'                                                                                                           )
    print('        proconsul_top_k          : Length of the pvalues subset for the PROCONSUL top-k sampling. If 0 no top-k sampling.'                                                 )
    print('                                   If you insert a list of values, multiple version of PROCONSUL will be run. One for each value.'                           )
    print('                                   (default: 0)'                                                                                                             )
    print('                                                                                                                                                         '   )

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set disease, algorithms and validation')
    parser.add_argument('-a','--algs', nargs='+', default=["diamond", "proconsul"],
                    help='List of algorithms to run to collect results. (default: ["diamond", "proconsul"])')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation on which test the algorithms. It can be: "kfold", "extended" or "all". If all, perform both the validations. (default: all)')
    parser.add_argument('--disease_file', type=str, default="data/diamond_dataset/diseases.txt",
                    help='Relative path to the file containing the disease names to use for the comparison. (default: "data/diamond_dataset/diseases.txt")')
    parser.add_argument('--database', type=str, default="diamond_dataset",
                    help='Database name from which take the PPIs. Choose from "biogrid", "stringdb", "pnas", or "diamond_dataset". (default: "diamond_dataset")')
    parser.add_argument('--proconsul_n_rounds', type=int, nargs='+', default=[10],
                    help='How many different rounds PROCONSUL will do to reduce statistical fluctuation. If you insert a list of values multiple version of PROCONSUL will be run. One for each value. (default: 10)')
    parser.add_argument('--proconsul_temp', type=float, nargs='+', default=[1.0],
                    help='Temperature value for the pDIAMOnD softmax function. If you insert a list of values, multiple version of PROCONSUL will be run. One for each value. (default: 1.0)')
    parser.add_argument('--proconsul_top_p', type=float, nargs='+', default=[0.0],
                    help='Probability threshold value for pDIAMOnD nucleus sampling. If 0 no nucleus sampling. If you insert a list of values, multiple version of PROCONSUL will be run. One for each value. (default: 0.0)')
    parser.add_argument('--proconsul_top_k', type=int, nargs='+', default=[0],
                    help='Length of the pvalues subset for Top-K sampling. If 0 no top-k sampling. If you insert a list of values, multiple version of PROCONSUL will be run. One for each value. (default: 0)')
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

    # 0. Read parsed values
    algs                = args.algs
    validation          = args.validation
    disease_file        = args.disease_file
    database_name       = args.database
    proconsul_n_rounds  = args.proconsul_n_rounds
    proconsul_temp      = args.proconsul_temp
    proconsul_top_p     = args.proconsul_top_p
    proconsul_top_k     = args.proconsul_top_k

    # 1. Check algorithm names
    for alg in algs:

        if alg not in ["diamond", "proconsul"]:
            print(f"ERROR: {alg} is not a valid algorithm!")
            print_usage()
            sys.exit(0)

    # 2. Check validation
    if validation not in ["kfold", "extended", "all"]:
        print(f"ERROR: {validation} is no valid validation method!")
        print_usage()
        sys.exit(1)

    if validation == 'all':
        validations = ['kfold', 'extended']
    else:
        validations = [validation]

    # 3. Get diesease list from file
    try:
        diseases = read_disease_file(disease_file)
    except:
        print(f"Not found file in {disease_file} or no valid location.")
        print_usage()
        sys.exit(1)

    # if empty, exit with error
    if len(diseases) == 0:
        print(f"ERROR: No diseases in disease_file")
        sys.exit(1)

    # 4. Check database
    if database_name not in ["biogrid", "stringdb", "pnas", "diamond_dataset"]:
        print("ERROR: no valid database name")
        print_usage()
        sys.exit(1)

    if database_name == "biogrid":
        database_path = "data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"

    if database_name == "stringdb":
        database_path = "data/9606.protein.links.full.v11.5.txt"

    if database_name == "pnas":
        database_path = "data/pnas.2025581118.sd02.csv"
    
    if database_name == "diamond_dataset":
        database_path = "data/diamond_dataset/Interactome.tsv"

    # 5. Check PROCONSUL number of round
    for round in proconsul_n_rounds:
        if round <= 0:
            print(f"ERROR: The number of rounds must be greater or equal 1.")
            print_usage()
            sys.exit(1)

    # 6. Check PROCONSUL temperatures
    for temp in proconsul_temp:
        if temp < 0:
            print(f"ERROR: The temperature must be greater or equal 0.")
            print_usage()
            sys.exit(1)

    # 7. Check PROCONSUL top-p sampling
    for top_p in proconsul_top_p:
        if top_p < 0 or top_p > 1:
            print("ERROR: top_p must be in [0,1]")
            print_usage()
            sys.exit(1)

    # 8. Check PROCONSUL top-k sampling
    for top_k in proconsul_top_k:
        if top_k < 0:
            print("ERROR: top_k must be greater or equal 0.")
            print_usage()
            sys.exit(1)
    
    # 9. Build the schedules of PROCONSUL hyperparameters
    max_len = max([len for len in [len(proconsul_n_rounds), len(proconsul_temp), len(proconsul_top_p), len(proconsul_top_k)]])
    
    # 9.1. n_rounds
    len_n_rounds = len(proconsul_n_rounds)
    if len_n_rounds < max_len:
        for i in range(max_len - len_n_rounds):
            proconsul_n_rounds.append(10)   # append the deafault value to fill
                                            # the gap with the other hyperparams
    # 9.2. temperature
    len_temperatures = len(proconsul_temp)
    if len_temperatures < max_len:
        for i in range(max_len - len_temperatures):
            proconsul_temp.append(1.0)

    # 9.3. top_p
    len_top_p = len(proconsul_top_p)
    if len_top_p < max_len:
        for i in range(max_len - len_top_p):
            proconsul_top_p.append(0.0)

    # 9.4. top_k
    len_top_k = len(proconsul_top_k)
    if len_top_k < max_len:
        for i in range(max_len - len_top_k):
            proconsul_top_k.append(0)
    
    # 9.5. final check to see if all the hyperparameters have the same length
    assert(len(proconsul_n_rounds) == len(proconsul_temp) == len(proconsul_top_k) == len(proconsul_top_p))

    # 10. Make a list of tuples [(alg_name, hyperparams)]
    algs_and_hyperparams = []

    for alg in algs:
        if alg == "diamond":
            algs_and_hyperparams.append((alg, {}))
        
        if alg == "proconsul":
            proconsul_n_instances = len(proconsul_n_rounds)
            for i in range(proconsul_n_instances):
                algs_and_hyperparams.append((alg, {"proconsul_n_rounds": proconsul_n_rounds[i],
                                                   "proconsul_temp": proconsul_temp[i],
                                                   "proconsul_top_p": proconsul_top_p[i],
                                                   "proconsul_top_k": proconsul_top_k[i]}))


    # 11. Print all the parsed inputs.
    print('                                                    ')
    print(f"===================================================")
    print(f"Algorithms: {algs}"                                 )
    print(f"Validations: {validations}"                         )
    print(f"Diseases: {len(diseases)}"                          )
    print(f"Database name: {database_name}"                     )
    print(f"Database path: {database_path}"                     )
    print(f"PROCONSUL number of rounds: {proconsul_n_rounds}"   )
    print(f"PROCONSUL temperatures: {proconsul_temp}"           )
    print(f"PROCONSUL top-p: {proconsul_top_p}"                 )
    print(f"PROCONSUL top-k: {proconsul_top_k}"                 )
    print(f"===================================================")
    print('                                                    ')

    return algs_and_hyperparams, validations, diseases, database_name, database_path


# main
if __name__ == "__main__":

    # Read input
    args = parse_args()
    algs_and_hyperparams, validations, diseases, database_name, database_path = read_terminal_input(args)

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

    # Print information
    # print(f"{database_name.upper()} HHI:")
    # print(nx.info(hhi))
    # print(" ")
    # print(f"{database_name.upper()} HHI LLC:")
    # print(nx.info(hhi_lcc))
    # print(" ")

    # -------------------------------
    #     K-FOLD CROSS VALIDATION
    # -------------------------------

    gda_curated = "data/curated_gene_disease_associations.tsv"
    seeds_file = "data/diamond_dataset/seeds.tsv"

    if 'kfold' in validations:
        for a_and_h in algs_and_hyperparams:
            
            # Separate algorithm name and its hyperparameters.
            alg = a_and_h[0]
            hyperparams = a_and_h[1]

            for disease in diseases:

                if database_name == "diamond_dataset":
                    disease_genes = get_disease_genes_from_seeds_file(seeds_file, disease, fix_random=True)
                else:
                    disease_genes = get_disease_genes_from_gda(gda_curated, disease, translate_in=database_name, training_mode=True)
                
                # check that the list of disease genes is not empty
                if len(disease_genes) == 0:
                    print(f"WARNING: {disease} has no disease genes. Skip this disease")
                    continue

                k_fold_cross_validation(hhi_lcc,
                                        alg,
                                        disease,
                                        disease_genes,
                                        K=5,
                                        database_name=database_name,
                                        hyperparams=hyperparams)

    # Skip the extended validation if we are using the DIAMOnD Dataset.
    if database_name == "diamond_dataset":
        print("No extended validation for the diamond_dataset.")
        print("Done!")
        sys.exit(0)

    # --------------------------------
    #       EXTENDED VALIDATION
    # --------------------------------

    gda_all = "data/all_gene_disease_associations.tsv"

    if 'extended' in validations:
        
        for a_and_h in algs_and_hyperparams:
            
            # Separate algorithm name and its hyperparameters.
            alg = a_and_h[0]
            hyperparams = a_and_h[1]

            for disease in diseases:

                # get disease genes from curated and all GDA
                curated_disease_genes = get_disease_genes_from_gda(gda_curated, disease, translate_in=database_name, training_mode=True)
                all_disease_genes = get_disease_genes_from_gda(gda_all, disease, translate_in=database_name, training_mode=True)

                # remove from all the genes that are already in curated
                extended_genes = list(set(all_disease_genes) - set(curated_disease_genes))

                # check if the disease genes lists are not empty
                if len(curated_disease_genes) == 0:
                    print(f"WARNING: {disease} has no curated disease genes. Skip this disease")
                    continue

                if len(all_disease_genes) == 0:
                    print(f"WARNING: {disease} has no all disease genes. Skip this disease")
                    continue

                if len(extended_genes) == 0:
                    print(f"WARNING: {disease} has no extended disease genes. Skip this disease")
                    continue

                # run the extended validation on <algorithm>
                extended_validation(hhi_lcc,
                                    alg,
                                    disease,
                                    curated_disease_genes,
                                    extended_genes,
                                    database_name=database_name,
                                    hyperparams=hyperparams)
    print("Done!")
