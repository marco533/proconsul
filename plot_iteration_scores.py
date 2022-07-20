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
import seaborn as sns

from utils.data_utils import *
from utils.network_utils import *

# ===========================
#     R E A D   I N P U T
# ===========================

def print_usage():
    print(' ')
    print('        usage: python3 plot_iteration_scores.py --algs --metric --validation --disease_file --database --heat_diffusion_time --pdiamond_n_iters --pdiamond_temp --pdiamond_top_p --pdiamond_top_k')
    print('        -----------------------------------------------------------------')
    print('        algs                     : List of algorithms to use to collect results.')
    print('                                   They can be: "diamond", "diamond2, "pdiamond", "pdiamond_log", "pdiamond_entropy", "heat_diffusion"')
    print('                                   If all, run all the algorithms. (default: all')
    print('        metric                   : Metric to use to plot the scores. It can be')
    print('                                   "precision", "recall", "f1" or "ndcg".')
    print('                                   If all, makes a plot for each metric. (default: all')
    print('        validation               : Type of validation on which test the algorithms. It can be')
    print('                                   "kfold", "extended" or "all".')
    print('                                    If all, perform both the validations. (default: all')
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison')
    print('                                   (default: "data/disease_file.txt).')
    print('        database                 : Database name from which take the PPIs. Choose from "biogrid", "stringdb", "pnas", or "diamond_dataset".')
    print('                                   (default: "biogrid)')
    print('        heat_diffusion_time      : Diffusion time for Heat Diffusion algorithm.')
    print('                                   (default: 0.005)')
    print('        pdiamond_n_iters         : Number of iteration for pDIAMOnD.')
    print('                                   (default: 10)')
    print('        pdiamond_temp            : Temperature value for the pDIAMOnD softmax function.')
    print('                                   (default: 1.0)')
    print('        pdiamond_top_p           : Probability threshold value for pDIAMOnD nucleus sampling. If 0 no nucleus sampling')
    print('                                   (default: 0.0)')
    print('        pdiamond_top_k           : Length of the pvalues subset for Top-K sampling. If 0 no top-k sampling')
    print('                                   (default: 0)')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    parser = argparse.ArgumentParser(description='Set disease, algorithms and validation')
    parser.add_argument('-a','--algs', nargs='+', default=["diamond", "pdiamond", "pdiamond_log", "heat_diffusion"],
                    help='List of algorithms to run (default: all)')
    parser.add_argument('--metric', type=str, default='recall',
                    help='Metric to use for make the plot. (default: recall')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation. (default: all')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Relative path to the file with disease names (default: "data/disease_file.txt)')
    parser.add_argument('--database', type=str, default="biogrid",
                    help='Database name (default: "biogrid')
    parser.add_argument('--heat_diffusion_time', type=float, default=0.005,
                    help='Diffusion time for heat_diffusion algorithm. (default: 0.005')
    parser.add_argument('--pdiamond_n_iters', type=int, default=10,
                    help='Number of iteration for pDIAMOnD. (default: 10)')
    parser.add_argument('--pdiamond_temp', type=float, default=1.0,
                    help='Temperature value for the pDIAMOnD softmax function. (default: 1.0)')
    parser.add_argument('--pdiamond_top_p', type=float, default=0.0,
                    help='Probability threshold value for pDIAMOnD nucleus sampling. (default: 0.0)')
    parser.add_argument('--pdiamond_top_k', type=int, default=0,
                    help='Length of the pvalues subset for Top-K sampling. (default: 0)')
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
    metric              = args.metric
    validation          = args.validation
    disease_file        = args.disease_file
    database            = args.database
    heat_diffusion_time = args.heat_diffusion_time
    pdiamond_n_iters    = args.pdiamond_n_iters
    pdiamond_temp       = args.pdiamond_temp
    pdiamond_top_p      = args.pdiamond_top_p
    pdiamond_top_k      = args.pdiamond_top_k

    # 1. Check algorithm names
    # for alg in algs:
    #     if alg not in ["diamond", "pdiamond_log", "pdiamond_t0.5", "pdiamond_t1", "pdiamond_t10"]:
    #         print(f"ERROR: {alg} is not a valid algorithm!")
    #         print_usage()
    #         sys.exit(0)

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

    # Get list of validations
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

    # 6. Check diffusion time
    if heat_diffusion_time < 0:
        print(f"ERROR: diffusion time must be greater than 0")
        print_usage()
        sys.exit(1)

    # 7. Check pDIAMOnD number of iterations
    if pdiamond_n_iters <= 0:
        print(f"ERROR: pdiamond_n_iters must be greater or equal 1")
        print_usage()
        sys.exit(1)

    # 8. Check pDIAMOnD temperature
    if pdiamond_temp < 0:
        print(f"ERROR: pdiamond_temp must be greater or equal 0")
        print_usage()
        sys.exit(1)

    # 9. Check pDIAMOnD top-p sampling
    if pdiamond_top_p < 0 or pdiamond_top_p > 1:
        print("ERROR: top_p must be in [0,1]")
        print_usage()
        sys.exit(1)

    # 10. Check pDIAMOnD top-k sampling
    if pdiamond_top_k < 0:
        print("ERROR: pdiamond_top_k must be greater or equal 0")
        print_usage()
        sys.exit(1)

    print('                                                    ')
    print(f"===================================================")
    print(f"Algorithms: {algs}"                                 )
    print(f"Metric: {metric}"                         )
    print(f"Validations: {validations}"                         )
    print(f"Diseases: {len(diseases)}"                          )
    print(f"Database: {database}"                               )
    print(f"Heat Diffusion Time: {heat_diffusion_time}"         )
    print(f"pDIAMOnD number of iterations: {pdiamond_n_iters}"  )
    print(f"pDIAMOnD temperature: {pdiamond_temp}"  )
    print(f"pDIAMOnD top-p: {pdiamond_top_p}"                   )
    print(f"pDIAMOnD top-k: {pdiamond_top_k}"                   )
    print(f"===================================================")
    print('                                                    ')


    return algs, metric, validations, diseases, database, database_path, heat_diffusion_time, pdiamond_n_iters, pdiamond_temp, pdiamond_top_p, pdiamond_top_k


# main
if __name__ == "__main__":

    # Read input
    args = parse_args()
    algs, metric, validations, diseases, database_name, database_path, heat_diffusion_time, pdiamond_n_iters, pdiamond_temp, pdiamond_top_p, pdiamond_top_k = read_terminal_input(args)

    # Compact all the algorithm hyperparameters in a dictionary
    hyperparams = {"heat_diffusion_time": heat_diffusion_time,
                   "pdiamond_n_iters": pdiamond_n_iters,
                   "pdiamond_temp": pdiamond_temp,
                   "pdiamond_top_p": pdiamond_top_p,
                   "pdiamond_top_k": pdiamond_top_k}

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

    gda_curated = "data/curated_gene_disease_associations.tsv"
    seeds_file = "data/diamond_dataset/seeds.tsv"

    if 'kfold' in validations:
        for disease in diseases:
            data = []

            for alg in algs:
                alg_real_name = alg
                
                if alg == "proconsul_t0.5":
                    hyperparams["pdiamond_temp"] = 0.5
                    alg_real_name = "pdiamond_log"
                if alg == "proconsul_t1":
                    hyperparams["pdiamond_temp"] = 1
                    alg_real_name = "pdiamond_log"
                if alg == "proconsul_t10":
                    hyperparams["pdiamond_temp"] = 10
                    alg_real_name = "pdiamond_log"

                if database_name in ["diamond_dataset"]:
                    disease_genes = get_disease_genes_from_seeds_file(seeds_file, disease, fix_random=True)
                else:
                    disease_genes = get_disease_genes_from_gda(gda_curated, disease, translate_in=database_name)
                
                # check that the list of disease genes is not empty
                if len(disease_genes) == 0:
                    print(f"WARNING: {disease} has no disease genes. Skip this disease")
                    continue

                alg_scores = k_fold_cross_validation(hhi_lcc,
                                                    alg_real_name,
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
                
                # print(f"complete k-fold scores of {metric}: \n{alg_scores}")

                # Fill the DataFrame
                for i in range(len(alg_scores)):
                    data.append(
                        {
                            metric: alg_scores[i],
                            'algorithms': alg,
                            'iterations': i
                        }
                    )

            disease_df = pd.DataFrame(data)

            # Plot
            plot = sns.lineplot(data=disease_df, x="iterations", y=metric, hue="algorithms").set_title(disease)
            fig = plot.get_figure()
            fig.savefig(f'plots/iteration_scores/{disease}.png', bbox_inches='tight')

            plt.close('all')

                    