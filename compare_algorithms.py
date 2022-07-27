import argparse
import sys
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from utils.network_utils import *
from utils.data_utils import *

# ===========================
#     R E A D   I N P U T
# ===========================

def print_usage():

    print(' ')
    print('        usage: python3 compare_algs.py --left_ring --right_ring --metric --validation --disease_file --database --heat_diffusion_time --pdiamond_n_iters --pdiamond_temp --pdiamond_top_p --pdiamond_top_k')
    print('        -----------------------------------------------------------------')
    print('        left_ring                : Algorithms in the LEFT part of the ring. We will compare the greatest algorithm on the left')
    print('                                   against the greatest algorithms on the right.')
    print('                                   They can be: "diamond", "proconsul_t*", where * is any temperature value.')
    print('        right_ring               : Algorithms in the RIGHT part of the ring. We will compare the greatest algorithm on the left')
    print('                                   against the greatest algorithms on the right.')
    print('                                   They can be: "diamond", "proconsul_t*", where * is any temperature value.')
    print('        left_name                : Name for the algorithms in the left ring.')
    print('        right_name               : Name for the algorithms in the right ring.')
    print('        metric                   : Metric to use to plot the scores. It can be')
    print('                                   "precision", "recall", "f1" or "ndcg".')
    print('        validation               : Type of validation on which test the algorithms. It can be')
    print('                                   "kfold", "extended" or "all".')
    print('                                   If all, perform both the validations. (default: all')
    print('        disease_file             : Relative path to the file containing the disease names to use for the comparison')
    print('                                   (default: "data/disease_file.txt).')
    print('        database                 : Database name from which take the PPIs. Choose from "biogrid", "stringdb", "pnas", or "diamond_dataset".')
    print('                                   (default: "diamond_dataset)')
    print('        proconsul_n_iters        : Number of iteration for PROCONSUL.')
    print('                                   (default: 10)')
    print(' ')

def parse_args():
    '''
    Parse the terminal arguments.
    '''
    
    parser = argparse.ArgumentParser(description='Set disease, algorithms and validation')
    
    parser.add_argument('-lr','--left_ring', nargs='+',
                    help='List of algorithms in the left part of the ring (default: all)')
    parser.add_argument('-rr','--right_ring', nargs='+',
                    help='List of algorithms in the right part of the ring (default: all)')
    parser.add_argument('-ln','--left_name', type=str, default="DIAMOnD",
                    help='Name for the algorithms in the left ring. (default: DIAMOnD)')
    parser.add_argument('-rn','--right_name', type=str, default="PROCONSUL",
                    help='Name for the algorithms in the right ring. (default: PROCONSUL)')
    parser.add_argument('--metric', type=str, default='f1',
                    help='Metric to use for make the plot. (default: f1')
    parser.add_argument('--validation', type=str, default='all',
                    help='Type of validation. (default: all')
    parser.add_argument('--disease_file', type=str, default="data/disease_file.txt",
                    help='Relative path to the file with disease names (default: "data/disease_file.txt)')
    parser.add_argument('--database', type=str, default="biogrid",
                    help='Database name (default: "biogrid')
    parser.add_argument('--proconsul_n_iters', type=int, default=10,
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

    # Read parsed values
    left_ring           = args.left_ring
    right_ring          = args.right_ring
    left_name           = args.left_name
    right_name          = args.right_name
    metric              = args.metric
    validation          = args.validation
    disease_file        = args.disease_file
    database            = args.database
    proconsul_n_iters   = args.proconsul_n_iters

    # # 1. Check algorithm names
    # for alg in algs:
    #     if alg not in ["diamond", "diamond2", "pdiamond", "pdiamond_log", "pdiamond_entropy", "heat_diffusion"]:
    #         print(f"ERROR: {alg} is not a valid algorithm!")
    #         print_usage()
    #         sys.exit(0)

    # 1. Metric
    if metric not in ["precision", "recall", "f1", "ndcg"]:
        print(f"ERROR: {metric} is not a valid metric!")
        print_usage()
        sys.exit(1)

    # 2. Check validation
    if validation not in ["kfold", "extended", "all"]:
        print(f"ERROR: {validation} is no valid validation method!")
        print_usage()
        sys.exit(1)

    # Get list of validations
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

    # 5. Check pDIAMOnD number of iterations
    if proconsul_n_iters <= 0:
        print(f"ERROR: pdiamond_n_iters must be greater or equal 1")
        print_usage()
        sys.exit(1)

    print('                                                    ')
    print(f"===================================================")
    print(f"{left_name} ring: {left_ring}"                      )
    print(f"{right_name} ring: {right_ring}"                    )
    print(f"Metric: {metric}"                                   )
    print(f"Validations: {validations}"                         )
    print(f"Diseases: {len(diseases)}"                          )
    print(f"Database: {database}"                               )
    print(f"PROCONSUL number of iterations: {proconsul_n_iters}")
    print(f"===================================================")
    print('                                                    ')


    return left_ring, right_ring, left_name, right_name, metric, validations, diseases, database, database_path, proconsul_n_iters


def get_score(algorithm=None, disease=None, database=None, validation=None, K=None, metric=None, diffusion_time=None, n_iters=None, temp=None, top_p=None, top_k=None):
    """
    Get the algorithm score on a given disease.
    """


    # Get the relative path to the algorithm score
    score_path = f"results/{database}/{validation}/{algorithm}/{algorithm}__{string_to_filename(disease)}"

    if diffusion_time is not None:
        return
    if n_iters is not None:
        score_path += f"__{n_iters}_iters"
    if temp is not None:
        score_path += f"__temp_{temp}"
    if top_p is not None:
        score_path += f"__top_p_{top_p}"
    if top_k is not None:
        score_path += f"__top_k_{top_k}"
    if validation == "extended":
        score_path += f"__{validation}.csv"
    if validation == "kfold":
        score_path += f"__{K}_fold.csv"

    # Read the CSV score file as a DataFrame
    scores_df = pd.read_csv(score_path, index_col=0)

    # If metric is not None, select only the results relative to that metric
    if metric:
        indices = scores_df.index.to_list()
        indices_to_remove = []
        for idx in indices:
            if idx != metric:
                indices_to_remove.append(idx)
        
        # Drop out the other metrics
        scores_df = scores_df.drop(indices_to_remove)

    if validation == "extended":
        # Return the array with the scores
        return scores_df.to_numpy()

    if validation == "kfold":
        # Here we need to convert the strings to float values
        scores_to_fix = scores_df.to_numpy()

        scores = np.zeros((scores_to_fix.shape[0], scores_to_fix.shape[1]), dtype=np.float)
        scores_std = np.zeros((scores_to_fix.shape[0], scores_to_fix.shape[1]), dtype=np.float)

        for i, size in enumerate(scores_to_fix):
            for j, value in enumerate(size):
                value = value.replace("(", "")
                value = value.replace(")", "")
                score, std = value.split(", ")
                score = float(score)
                std = float(std)

                scores[i][j] = score
                scores_std[i][j] = std

        return scores, scores_std


def compare_algs_heatmap(left_ring=None, right_ring=None, left_name=None, right_name=None,
                         metric=None, validation=None, diseases=None, proconsul_n_iters=None,
                         database_name=None):

    """
    Heatmap that, for each number of predicted genes compare the best algorithm
    in "left_ring" with the best algorithm in "right_ring".
    """
    data = {}

    if validation == "kfold":
        columns = ["KF Top 25", "KF Top 50", "KF Top 100", "KF Top 200"]
    
    if validation == "extended":
        columns = ["EX Top 25", "EX Top 50", "EX Top 100", "EX Top 200"]

    for disease in diseases:
        best_left_scores = np.zeros(len(columns))
        best_right_scores = np.zeros(len(columns))

        # Get the scores for each algorithm in the right ring.
        # When one value is highet than the currespondent value in
        # best_right_scores, substitute it.
        for ra in right_ring:
            if "proconsul" in ra:
                alg, temp = ra.split("_t")
                temp = float(temp)
                alg = "pdiamond_log"
                scores, _ = get_score(algorithm=alg,
                                      disease=disease,
                                      database=database_name, 
                                      metric=metric,
                                      validation=validation,
                                      K=5,
                                      n_iters=proconsul_n_iters,
                                      temp=temp, top_k=0, top_p=0.0)
            else:
                alg = ra
                scores, _ = get_score(algorithm=alg,
                                      disease=disease,
                                      database=database_name,
                                      metric=metric,
                                      validation=validation,
                                      K=5)
            
            scores = scores[0]

            print(f"right scores: {scores}")
            for idx, score in enumerate(scores):
                if score > best_right_scores[idx]:
                    best_right_scores[idx] = score
        
        print(f"best right scores: {best_right_scores}")

        print(" ")

        # Do the same for the left ring algorithms
        for la in left_ring:
            if "proconsul" in la:
                alg, temp = la.split("_t")
                temp = float(temp)
                alg = "pdiamond_log"
                scores, _ = get_score(algorithm=alg, 
                                      disease=disease,
                                      database=database_name,
                                      metric=metric,
                                      validation=validation,
                                      K=5,
                                      n_iters=proconsul_n_iters,
                                      temp=temp, top_k=0, top_p=0.0)
            else:
                alg = la     
                scores, _ = get_score(algorithm=alg,
                                      disease=disease,
                                      database=database_name,
                                      metric=metric,
                                      validation=validation,
                                      K=5)

            
            scores = scores[0]                          
            
            print(f"left scores: {scores}")
            for idx, score in enumerate(scores):
                if score > best_left_scores[idx]:
                    best_left_scores[idx] = score
        
        print(f"best lest scores: {best_left_scores}")

        # Compare left and right scores
        compared_scores = best_left_scores - best_right_scores

        print(f"compared scores: {compared_scores}")

        data[disease] = compared_scores

    print(f"data:\n{data}")

    # Build the DataFrame using 'data'
    df = pd.DataFrame.from_dict(data=data, orient='index', columns=columns)
    print(df.head())

    # From the DataFrame build the HeatMap

    sns.set(rc = {'figure.figsize':(15,15)})
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    hm = sns.heatmap(df, annot=False, cmap=cmap, center=0, cbar_kws={'label': f'<-- {right_name} | {left_name} -->'})

    fig = hm.get_figure()
    fig.suptitle(f"{left_name} VS {right_name} on {metric.upper()} Score")
    fig.savefig(f'plots/heatmaps/{left_name}_vs_{right_name}_{database_name}.png', bbox_inches='tight')

    # Close previous plots
    plt.close()



if __name__ == "__main__":
    
    # Read input
    args = parse_args()
    left_ring, right_ring, left_name, right_name, metric, validations, diseases, database_name, database_path, proconsul_n_iters = read_terminal_input(args)

    # Compact all the algorithm hyperparameters in a dictionary
    hyperparams = {"heat_diffusion_time": 0,
                   "pdiamond_n_iters": proconsul_n_iters,
                   "pdiamond_temp": 0,
                   "pdiamond_top_p": 0.0,
                   "pdiamond_top_k": 0}

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

    for validation in validations:
        compare_algs_heatmap(left_ring=left_ring, right_ring=right_ring, left_name=left_name, right_name=right_name,
                             metric=metric, validation=validation, diseases=diseases, proconsul_n_iters=proconsul_n_iters,
                             database_name=database_name)