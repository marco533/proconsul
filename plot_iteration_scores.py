# =====================
#     I M P O R T S
# =====================

import argparse
import sys
from test.extended_validation import *
from test.kfold_validation import *

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.data_utils import *
from utils.network_utils import *

# ===========================
#     R E A D   I N P U T
# ===========================

def print_usage():
    print(' ')
    print('        usage: python3 plot_iteration_scores.py --algs --metric --validation --disease_file --database --proconsul_n_rounds --proconsul_temp --proconsul_top_p --proconsul_top_k')
    print('        -----------------------------------------------------------------')
    print('        algs                     : List of algorithms for which plot the iteration scores.')
    print('                                   They can be: "diamond" or "proconsul" (default: all)')
    print('        metric                   : Metric to use to plot the scores. It can be')
    print('                                   "precision", "recall", "f1" or "ndcg".')
    print('                                   If all, makes a plot for each metric. (default: all')
    print('        validation               : Type of validation on which test the algorithms. It can be')
    print('                                   "kfold", "extended" or "all".')
    print('                                   If all, perform both the validations. (default: all)')
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison.')
    print('                                   (default: "data/diamond_dataset/diseases.txt).')
    print('        database                 : Database name from which take the PPIs. Choose from "biogrid", "stringdb", "pnas", or "diamond_dataset".')
    print('                                   (default: "diamond_dataset)')
    print('        proconsul_n_rounds       : How many different rounds PROCONSUL will do to reduce statistical fluctuation.')
    print('                                   If you insert a list of values multiple version of PROCONSUL will be run. One for each value.')
    print('                                   (default: 10)')
    print('        proconsul_temp           : Temperature value for the PROCONSUL softmax function.')
    print('                                   If you insert a list of values, multiple version of PROCONSUL will be run. One for each value.')
    print('                                   (default: 1.0)')
    print('        proconsul_top_p          : Probability threshold value for PROCONSUL nucleus sampling. If 0 no nucleus sampling')
    print('                                   If you insert a list of values, multiple version of PROCONSUL will be run. One for each value.')
    print('                                   (default: 0.0)')
    print('        proconsul_top_k          : Length of the pvalues subset for the PROCONSUL top-k sampling. If 0 no top-k sampling.')
    print('                                   If you insert a list of values, multiple version of PROCONSUL will be run. One for each value.')
    print('                                   (default: 0)')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set disease, algorithms and validation')
    parser.add_argument('-a','--algs', nargs='+', default=["diamond", "proconsul"],
                    help='List of algorithms to run to collect results. (default: ["diamond", "proconsul"])')
    parser.add_argument('--metric', type=str, default='recall',
                    help='Metric to use for make the plot. (default: recall')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation on which test the algorithms. It can be: "kfold", "extended" or "all". If all, perform both the validations. (default: all)')
    parser.add_argument('--disease_file', type=str, default="data/diamond_dataset/diseases.txt",
                    help='Relative path to the file containing the disease names to use for the comparison. (default: "data/diamond_dataset/diseases.txt")')
    parser.add_argument('--database', type=str, default="diamond_dataset",
                    help='Database name from which take the PPIs. Choose from "biogrid", "stringdb", "pnas", or "diamond_dataset". (default: "diamond_dataset")')
    parser.add_argument('--proconsul_n_rounds', type=int, nargs='+', default=[10],
                    help='How many different rounds PROCONSUL will do to reduce statistical fluctuation. If you insert a list of values multiple version of PROCONSUL will be run. One for each value. (default: 10)')
    parser.add_argument('--proconsul_temp', type=float, nargs='+', default=[1.0],
                    help='Temperature value for the PROCONSUL softmax function. If you insert a list of values, multiple version of PROCONSUL will be run. One for each value. (default: 1.0)')
    parser.add_argument('--proconsul_top_p', type=float, nargs='+', default=[0.0],
                    help='Probability threshold value for PROCONSUL nucleus sampling. If 0 no nucleus sampling. If you insert a list of values, multiple version of PROCONSUL will be run. One for each value. (default: 0.0)')
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
    metric              = args.metric
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

    # 2. Check metric
    if metric not in ["precision", "recall", "f1", "ndcg"]:
        print(f"ERROR: {metric} is no valid metric!")
        print_usage()
        sys.exit(1)

    # 3. Check validation
    if validation not in ["kfold", "extended", "all"]:
        print(f"ERROR: {validation} is no valid validation method!")
        print_usage()
        sys.exit(1)

    if validation == 'all':
        validations = ['kfold', 'extended']
    else:
        validations = [validation]

    # 4. Get diesease list from file
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

    # 5. Check database
    if database_name not in ["biogrid", "stringdb", "pnas", "diamond_dataset"]:
        print("ERROR: no valid database name")
        print_usage()
        sys.exit(1)

    if database_name == "biogrid":
        database_path = "data/biogrid/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt"

    if database_name == "stringdb":
        database_path = "data/stringdb/9606.protein.links.full.v11.5.txt"

    if database_name == "pnas":
        database_path = "data/pnas/pnas.2025581118.sd02.csv"
    
    if database_name == "diamond_dataset":
        database_path = "data/diamond_dataset/Interactome.tsv"

    # 6. Check PROCONSUL number of round
    for round in proconsul_n_rounds:
        if round <= 0:
            print(f"ERROR: The number of rounds must be greater or equal 1.")
            print_usage()
            sys.exit(1)

    # 7. Check PROCONSUL temperatures
    for temp in proconsul_temp:
        if temp < 0:
            print(f"ERROR: The temperature must be greater or equal 0.")
            print_usage()
            sys.exit(1)
        
        # If temp = 0 replace it with very samll number
        # to avoid nan values
        if temp == 0:
            temp = 1e-40

    # 8. Check PROCONSUL top-p sampling
    for top_p in proconsul_top_p:
        if top_p < 0 or top_p > 1:
            print("ERROR: top_p must be in [0,1]")
            print_usage()
            sys.exit(1)

    # 9. Check PROCONSUL top-k sampling
    for top_k in proconsul_top_k:
        if top_k < 0:
            print("ERROR: top_k must be greater or equal 0.")
            print_usage()
            sys.exit(1)
    
    # 10. Build the schedules of PROCONSUL hyperparameters
    max_len = max([len for len in [len(proconsul_n_rounds), len(proconsul_temp), len(proconsul_top_p), len(proconsul_top_k)]])
    
    # 10.1. n_rounds
    len_n_rounds = len(proconsul_n_rounds)
    if len_n_rounds < max_len:
        for i in range(max_len - len_n_rounds):
            proconsul_n_rounds.append(10)   # append the deafault value to fill
                                            # the gap with the other hyperparams
    # 10.2. temperature
    len_temperatures = len(proconsul_temp)
    if len_temperatures < max_len:
        for i in range(max_len - len_temperatures):
            proconsul_temp.append(1.0)

    # 10.3. top_p
    len_top_p = len(proconsul_top_p)
    if len_top_p < max_len:
        for i in range(max_len - len_top_p):
            proconsul_top_p.append(0.0)

    # 10.4. top_k
    len_top_k = len(proconsul_top_k)
    if len_top_k < max_len:
        for i in range(max_len - len_top_k):
            proconsul_top_k.append(0)
    
    # 10.5. final check to see if all the hyperparameters have the same length
    assert(len(proconsul_n_rounds) == len(proconsul_temp) == len(proconsul_top_k) == len(proconsul_top_p))

    # 11. Make a list of tuples [(alg_name, hyperparams)]
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
    print(f"Metric: {metric}"                                   )
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

    return algs_and_hyperparams, metric, validations, diseases, database_name, database_path


# main
if __name__ == "__main__":

    # Read input
    args = parse_args()
    algs_and_hyperparams, metric, validations, diseases, database_name, database_path = read_terminal_input(args)

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

    # -------------------------------
    #     K-FOLD CROSS VALIDATION
    # -------------------------------

    gda_curated = "data/gda/curated_gene_disease_associations.tsv"
    seeds_file = "data/diamond_dataset/seeds.tsv"

    if 'kfold' in validations:
        for disease in diseases:
            data = []

            for a_and_h in algs_and_hyperparams:
                
                # Separate algorithm name and its hyperparameters.
                alg = a_and_h[0]
                hyperparams = a_and_h[1]
                
                if database_name in ["diamond_dataset"]:
                    disease_genes = get_disease_genes_from_seeds_file(seeds_file, disease, fix_random=True)
                else:
                    disease_genes = get_disease_genes_from_gda(gda_curated, disease, translate_in=database_name)
                
                # check that the list of disease genes is not empty
                if len(disease_genes) == 0:
                    print(f"WARNING: {disease} has no disease genes. Skip this disease")
                    continue

                alg_scores = k_fold_cross_validation(hhi_lcc,
                                                    alg,
                                                    disease,
                                                    disease_genes,
                                                    K=5,
                                                    database_name=database_name,
                                                    hyperparams=hyperparams,
                                                    all_iterations=True)
                
                # get only the results of the specified metric and discard the others
                if metric == 'precision':
                    alg_scores = alg_scores[0]
                if metric == 'recall':
                    alg_scores = alg_scores[1]
                if metric == 'f1':
                    alg_scores = alg_scores[2]
                if metric == 'ndcg':
                    alg_scores = alg_scores[3]
                
                
                if alg == "diamond":
                    alg_label = alg
                if alg == "proconsul":
                    t = hyperparams["proconsul_temp"]
                    p = hyperparams["proconsul_top_p"]
                    k = hyperparams["proconsul_top_k"]

                    alg_label = f"{alg}_t{t}"

                    if p > 0:
                        alg_label += f"_top_p{p}"
                    
                    if k > 0:
                        alg_label += f"top_k{k}"

                # Fill the DataFrame
                for i in range(len(alg_scores)):
                    data.append(
                        {
                            metric: alg_scores[i],
                            'algorithms': alg_label,
                            'iterations': i
                        }
                    )

            disease_df = pd.DataFrame(data)

            # Plot
            plot = sns.lineplot(data=disease_df, x="iterations", y=metric, hue="algorithms").set_title(disease)
            fig = plot.get_figure()
            fig.savefig(f'plots/iteration_scores/{disease}.png', bbox_inches='tight')

            plt.close('all')

                    